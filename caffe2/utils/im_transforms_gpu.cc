#include "caffe2/utils/im_transforms.h"


namespace caffe2{



template<class Context>
int roll_weighted_die(const std::vector<float>& probabilities,Context* context)
{
	std::vector<float> cumulative;
	std::partial_sum(&probabilities[0], &probabilities[0] + probabilities.size(),
	                   std::back_inserter(cumulative));

	float val;
	math::RandUniform<float,Context>(1,static_cast<float>(0),cumulative.back(),&val,context);

 // LOG(INFO)<<"val = "<<val;
//  LOG(INFO)<<"return = "<<std::lower_bound(cumulative.begin(), cumulative.end(), val)
//            - cumulative.begin();
	  // Find the position within the sequence and add 1
	return (std::lower_bound(cumulative.begin(), cumulative.end(), val)
	          - cumulative.begin());
}


//template
//int roll_weighted_die(const std::vector<float>& probabilities,CPUContext* context);

template
int roll_weighted_die(const std::vector<float>& probabilities,CUDAContext* context);


template<class Context>
cv::Mat ApplyResize(const cv::Mat& in_img,
	const ResizeParam& param,Context* context) {
  cv::Mat out_img;
  const int new_height = param.height;
  const int new_width = param.width;
  
  int interp_mode = cv::INTER_LINEAR;
  vector<string>interp_mode_param = param.interp_mode;
  int num_interp_mode = interp_mode_param.size();
  if (num_interp_mode > 0) {
    
    vector<float> probs(num_interp_mode, 1.f / num_interp_mode);

    int prob_num = roll_weighted_die(probs,context);
    
    switch (interp_mode_param[prob_num][1]) {
      case 'R':
        interp_mode = cv::INTER_AREA;
        break;
      case 'U':
        interp_mode = cv::INTER_CUBIC;
        break;
      case 'I':
        interp_mode = cv::INTER_LINEAR;
        break;
      case 'E':
        interp_mode = cv::INTER_NEAREST;
        break;
      case 'A':
        interp_mode = cv::INTER_LANCZOS4;
        break;
      default:
        LOG(FATAL) << "Unknown interp mode.";
    }
    
  }
  
  string resize_mode = param.resize_mode;

  if(resize_mode=="WARP"){
    cv::resize(in_img, out_img, cv::Size(new_width, new_height), 0, 0,
                 interp_mode);
  }else{
    LOG(INFO) << "Unknown resize mode.";
  }
  
//  LOG(INFO)<<"num_interp_mode = "<<num_interp_mode<<", resize_mode = "<<resize_mode;
  return out_img;
}

// template
// cv::Mat ApplyResize(const cv::Mat& in_img,
//   const ResizeParam& param,CPUContext* context);

template
cv::Mat ApplyResize(const cv::Mat& in_img,
 const ResizeParam& param,CUDAContext* context);


template<class Context>
void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta,Context* context) {
  float prob;
  math::RandUniform<float,Context>(1,0.f,1.f,&prob,context);
  if (prob < brightness_prob) {
    CAFFE_ENFORCE_GE(brightness_delta, 0,"brightness_delta must be non-negative.") ;
    float delta;
    math::RandUniform<float,Context>(1,-brightness_delta,brightness_delta,&delta,context);
    AdjustBrightness(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

// template
// void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
//     const float brightness_prob, const float brightness_delta,CPUContext* context);

template
void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta,CUDAContext* context);


void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    in_img.convertTo(*out_img, -1, 1, delta);
  } else {
    *out_img = in_img;
  }
}

template<class Context>
void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper,Context* context) {
  float prob;
  math::RandUniform<float,Context>(1,0.f,1.f,&prob,context);
  if (prob < contrast_prob) {
    CHECK_GE(upper, lower) << "contrast upper must be >= lower.";
    CHECK_GE(lower, 0) << "contrast lower must be non-negative.";
    float delta;
    math::RandUniform<float,Context>(1,lower,upper,&delta,context);
    AdjustContrast(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

// template
// void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
//     const float contrast_prob, const float lower, const float upper,CPUContext* context);

template
void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
   const float contrast_prob, const float lower, const float upper,CUDAContext* context);

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img) {
  if (fabs(delta - 1.f) > 1e-3) {
    in_img.convertTo(*out_img, -1, delta, 0);
  } else {
    *out_img = in_img;
  }
}

template<class Context>
void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper,Context* context) {
  float prob;
  math::RandUniform<float,Context>(1,0.f,1.f,&prob,context);
  if (prob < saturation_prob) {
    CHECK_GE(upper, lower) << "saturation upper must be >= lower.";
    CHECK_GE(lower, 0) << "saturation lower must be non-negative.";
    float delta;
    math::RandUniform<float,Context>(1,lower,upper,&delta,context);
    AdjustSaturation(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

// template
// void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
//     const float saturation_prob, const float lower, const float upper,CPUContext* context);

template
void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
   const float saturation_prob, const float lower, const float upper,CUDAContext* context);


void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img) {
  if (fabs(delta - 1.f) != 1e-3) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the saturation.
    channels[1].convertTo(channels[1], -1, delta, 0);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

template<class Context>
void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta,Context* context) {
  float prob;
  math::RandUniform<float,Context>(1,0.f,1.f,&prob,context);
  if (prob < hue_prob) {
    CHECK_GE(hue_delta, 0) << "hue_delta must be non-negative.";
    float delta;
    math::RandUniform<float,Context>(1,-hue_delta,hue_delta,&delta,context);
    AdjustHue(in_img, delta, out_img);
  } else {
    *out_img = in_img;
  }
}

// template
// void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
//                const float hue_prob, const float hue_delta,CPUContext* context);

template
void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
              const float hue_prob, const float hue_delta,CUDAContext* context);


void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img) {
  if (fabs(delta) > 0) {
    // Convert to HSV colorspae.
    cv::cvtColor(in_img, *out_img, CV_BGR2HSV);

    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);

    // Adjust the hue.
    channels[0].convertTo(channels[0], -1, 1, delta);
    cv::merge(channels, *out_img);

    // Back to BGR colorspace.
    cvtColor(*out_img, *out_img, CV_HSV2BGR);
  } else {
    *out_img = in_img;
  }
}

template<class Context>
void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob,Context* context) {
  float prob;
  math::RandUniform<float,Context>(1,0.f,1.f,&prob,context);
  if (prob < random_order_prob) {
    // Split the image to 3 channels.
    vector<cv::Mat> channels;
    cv::split(*out_img, channels);
    CHECK_EQ(channels.size(), 3);

    // Shuffle the channels.
    std::random_shuffle(channels.begin(), channels.end());
    cv::merge(channels, *out_img);
  } else {
    *out_img = in_img;
  }
}


// template
// void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
//                          const float random_order_prob,CPUContext* context);

template
void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob,CUDAContext* context);


template<class Context>
cv::Mat ApplyDistort(const cv::Mat& in_img,
  const DistortParam& param,Context* context) {
  cv::Mat out_img = in_img;
  float prob;

  math::RandUniform<float,Context>(1,0.f,1.f,&prob,context);
  
  if (prob > 0.5) {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob,
                     param.brightness_delta,context);

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob,
                   param.contrast_lower, param.contrast_upper,context);

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob,
                     param.saturation_lower, param.saturation_upper,context);

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob, param.hue_delta,context);

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob,context);
  } else {
    // Do random brightness distortion.
    RandomBrightness(out_img, &out_img, param.brightness_prob,
                     param.brightness_delta,context);

    // Do random saturation distortion.
    RandomSaturation(out_img, &out_img, param.saturation_prob,
                     param.saturation_lower, param.saturation_upper,context);

    // Do random hue distortion.
    RandomHue(out_img, &out_img, param.hue_prob, param.hue_delta,context);

    // Do random contrast distortion.
    RandomContrast(out_img, &out_img, param.contrast_prob,
                   param.contrast_lower, param.contrast_upper,context);

    // Do random reordering of the channels.
    RandomOrderChannels(out_img, &out_img, param.random_order_prob,context);
  }
  

  return out_img;
}

// template
// cv::Mat ApplyDistort(const cv::Mat& in_img,
//   const DistortParam& param,CPUContext* context);

template
cv::Mat ApplyDistort(const cv::Mat& in_img,
  const DistortParam& param,CUDAContext* context);

template<class Context>
void DistortImage(const Datum& datum ,Datum* distort_datum,TransformParam param, Context* context){
  if(!param.distort_param.distort_valid){
    distort_datum->CopyFrom(datum);
    return ;
  }

  if(datum.encoded()){
    CAFFE_ENFORCE_EQ( !(param.force_color&&param.force_gray),1
      ,"cannot set both force_color and force_gray");
    cv::Mat cv_img;

    if(param.force_color || param.force_gray){
      cv_img = DecodeDatumToCVMat(datum,param.force_color);
    }else{
      cv_img = DecodeDatumToCVMatNative(datum);
    }

    cv::Mat distort_img = ApplyDistort(cv_img,param.distort_param,context);

    EncodeCVMatToDatum(distort_img,"jpg",distort_datum);

    distort_datum->set_label(datum.label());
  }else{
    LOG(ERROR)<<"Only support encoded datum now.";
  }
}


// template
// void DistortImage(const Datum& datum ,Datum* distort_datum,TransformParam param, CPUContext* context);

template
void DistortImage(const Datum& datum ,Datum* distort_datum,TransformParam param, CUDAContext* context);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum) {
  std::vector<uchar> buf;
  cv::imencode("."+encoding, cv_img, buf);
  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                              buf.size()));
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_encoded(true);
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}



void UpdateBBoxByResizePolicy(const ResizeParam& param,const int old_width,
  const int old_height,NormalizedBBox* bbox){
  float new_height = param.height;
  float new_width = param.width;
  float orig_aspect = static_cast<float>(old_width)/old_height;
  float new_aspect = new_width/new_height;


  float x_min = bbox->xmin()*old_width;
  float y_min = bbox->ymin()*old_height;
  float x_max = bbox->xmax()*old_width;
  float y_max = bbox->ymax()*old_height;

  float padding;

  if(param.resize_mode == "WARP"){
    x_min = std::max(0.f,x_min*new_width/old_width);
    x_max = std::min(new_width,x_max*new_width/old_width);
    y_min = std::max(0.f,y_min*new_height/old_height);
    y_max = std::max(new_height,y_max*new_height/old_height);
  }else{
    LOG(FATAL)<<"Unknown resize mode .";
  }

  bbox->set_xmin(x_min/new_width);
  bbox->set_ymin(y_min/new_height);
  bbox->set_xmax(x_max/new_width);
  bbox->set_ymax(y_max/new_height);

}

//template<>
void TransformAnnotation(const AnnotatedDatum& anno_datum,const bool do_resize,
  const NormalizedBBox& crop_bbox,const bool do_mirror,
  ::google::protobuf::RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
  TransformParam param){
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  if(anno_datum.type() == caffe::AnnotatedDatum_AnnotationType_BBOX){
    for(int g = 0;g < anno_datum.annotation_group_size();++g){
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      bool has_valid_annotation = false;
      for(int a = 0; a < anno_group.annotation_size();++a){
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        NormalizedBBox resize_bbox = bbox;
        if(do_resize && param.resize_param.resize_valid){
          CAFFE_ENFORCE_GT(img_height,0);
          CAFFE_ENFORCE_GT(img_width,0);
          UpdateBBoxByResizePolicy(param.resize_param,img_width,img_height,&resize_bbox);
        }
        if(param.emit_constraint.emit_valid && !MeetEmitConstraint(crop_bbox,resize_bbox,
          param.emit_constraint)){
          continue;
        }
        NormalizedBBox proj_bbox;
        if(ProjectBBox(crop_bbox,resize_bbox,&proj_bbox)){
          has_valid_annotation = true;
          Annotation* transformed_anno = transformed_anno_group.add_annotation();
          transformed_anno ->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if(do_mirror){
            float temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }

          if(do_resize){
            //Unimplement
          }

        }
      }
      if(has_valid_annotation){
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_group_all->Add()->CopyFrom(transformed_anno_group);
      }
    }
  }else{
    LOG(FATAL)<<"Unknown annotation type.";
  }
}

template<class Context>
void TransformAnnotation(const AnnotatedDatum& anno_datum,const bool do_resize,
  const NormalizedBBox& crop_bbox,const bool do_mirror,
  vector<AnnotationGroup>* transformed_anno_vec,
  TransformParam param,Context* context){
  const int img_height = anno_datum.datum().height();
  const int img_width = anno_datum.datum().width();
  transformed_anno_vec->clear();
  if(anno_datum.type() == caffe::AnnotatedDatum_AnnotationType_BBOX){
    for(int g = 0;g < anno_datum.annotation_group_size();++g){
      const AnnotationGroup& anno_group = anno_datum.annotation_group(g);
      AnnotationGroup transformed_anno_group;
      bool has_valid_annotation = false;
      for(int a = 0; a < anno_group.annotation_size();++a){
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        NormalizedBBox resize_bbox = bbox;
        if(do_resize && param.resize_param.resize_valid){
          CAFFE_ENFORCE_GT(img_height,0);
          CAFFE_ENFORCE_GT(img_width,0);
          UpdateBBoxByResizePolicy(param.resize_param,img_width,img_height,&resize_bbox);
        }
        if(param.emit_constraint.emit_valid && !MeetEmitConstraint(crop_bbox,resize_bbox,
          param.emit_constraint)){
          continue;
        }
        NormalizedBBox proj_bbox;
        if(ProjectBBox(crop_bbox,resize_bbox,&proj_bbox)){
          has_valid_annotation = true;
          Annotation* transformed_anno = transformed_anno_group.add_annotation();
          transformed_anno ->set_instance_id(anno.instance_id());
          NormalizedBBox* transformed_bbox = transformed_anno->mutable_bbox();
          transformed_bbox->CopyFrom(proj_bbox);
          if(do_mirror){
            float temp = transformed_bbox->xmin();
            transformed_bbox->set_xmin(1 - transformed_bbox->xmax());
            transformed_bbox->set_xmax(1 - temp);
          }

          if(do_resize){
            //Unimplement
          }

        }
      }
      if(has_valid_annotation){
        transformed_anno_group.set_group_label(anno_group.group_label());
        transformed_anno_vec->push_back(transformed_anno_group);
      }
    }
  }else{
    LOG(FATAL)<<"Unknown annotation type.";
  }
}


// template 
// void TransformAnnotation(const AnnotatedDatum& anno_datum,const bool do_resize,
//   const NormalizedBBox& crop_bbox,const bool do_mirror,
//   vector<AnnotationGroup>* transformed_anno_vec,
//   TransformParam param,CPUContext* context);

template 
void TransformAnnotation(const AnnotatedDatum& anno_datum,const bool do_resize,
  const NormalizedBBox& crop_bbox,const bool do_mirror,
  vector<AnnotationGroup>* transformed_anno_vec,
  TransformParam param,CUDAContext* context);

template<class Context>
void ExpandImage(const cv::Mat& img,const float expand_ratio,
  NormalizedBBox* expand_bbox,cv::Mat* expand_img,TransformParam param,Context* context){
  const int img_height = img.rows;
  const int img_width = img.cols;
  const int img_channels = img.channels();
  int height = static_cast<int>(img_height*expand_ratio);
  int width = static_cast<int>(img_width*expand_ratio);
  float h_off,w_off;

  math::RandUniform<float,Context>(1,0.f,static_cast<float>(height - img_height),&h_off,context);
  math::RandUniform<float,Context>(1,0.f,static_cast<float>(width - img_width),&w_off,context);
  h_off = floor(h_off);
  w_off = floor(w_off);
  expand_bbox->set_xmin(-w_off/img_width);
  expand_bbox->set_ymin(-h_off/img_height);
  expand_bbox->set_xmax((width - w_off)/img_width);
  expand_bbox->set_ymax((height - h_off)/img_height);

  expand_img->create(height,width,img.type());
  expand_img->setTo(cv::Scalar(0));

  const bool has_mean_file = 0;
  const bool has_mean_value = param.mean_value.size() > 0;
  if(has_mean_value){
    std::vector<float> mean_value = param.mean_value;
    CAFFE_ENFORCE_EQ(mean_value.size()==1||mean_value.size()==img_channels,true,
      "Specify either 1 mean_value or as many as channels");
    if(img_channels > 1 && mean_value.size()==1){
      for(int c=1;c<img_channels;++c){
        mean_value.push_back(mean_value[0]);
      }
    }
    vector<cv::Mat> channels(img_channels);
    cv::split(*expand_img,channels);
    for(int c=0;c<img_channels;++c){
      channels[c] = mean_value[c];
    }
    cv::merge(channels,*expand_img);
  }
  cv::Rect bbox_roi(w_off,h_off,img_width,img_height);
  img.copyTo((*expand_img)(bbox_roi));
}

// template
// void ExpandImage(const cv::Mat& img,const float expand_ratio,
//   NormalizedBBox* expand_bbox,cv::Mat* expand_img,TransformParam param,CPUContext* context);

template
void ExpandImage(const cv::Mat& img,const float expand_ratio,
  NormalizedBBox* expand_bbox,cv::Mat* expand_img,TransformParam param,CUDAContext* context);

template<class Context>
void ExpandImage(const Datum& datum,const float expand_ratio,NormalizedBBox* expand_bbox,
  Datum* expand_datum,TransformParam param,Context* context){
  if(datum.encoded()){
    CAFFE_ENFORCE_EQ( !(param.force_color&&param.force_gray),true,
      "cannot set both force_color and force_gray");
    cv::Mat cv_img;
    if(param.force_color||param.force_gray){
      cv_img = DecodeDatumToCVMat(datum,param.force_color);
    }else{
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    cv::Mat expand_img;
    ExpandImage(cv_img,expand_ratio,expand_bbox,&expand_img,param,context);
    EncodeCVMatToDatum(expand_img,"jpg",expand_datum);
    expand_datum->set_label(datum.label());
    return ;
  }else{
    if(param.force_color||param.force_gray){
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }


}


// template
// void ExpandImage(const Datum& datum,const float expand_ratio,NormalizedBBox* expand_bbox,
//   Datum* expand_datum,TransformParam param,CPUContext* context);

template
void ExpandImage(const Datum& datum,const float expand_ratio,NormalizedBBox* expand_bbox,
  Datum* expand_datum,TransformParam param,CUDAContext* context);

template<class Context>
void ExpandImage(const AnnotatedDatum& anno_datum,AnnotatedDatum* expand_anno_datum,
  TransformParam param,Context* context){
  if(!param.expand_param.expand_valid){
    expand_anno_datum->CopyFrom(anno_datum);
    return ;
  }
  const ExpandParam& expand_param = param.expand_param;
  const float expand_prob = expand_param.expand_prob;
  float prob;
  math::RandUniform(1,0.f,1.f,&prob,context);
  if(prob>expand_prob){
    expand_anno_datum->CopyFrom(anno_datum);
    return;
  }
  const float max_expand_ratio = expand_param.max_expand_ratio;
  if(fabs(max_expand_ratio-1.0)<1e-2){
    expand_anno_datum->CopyFrom(anno_datum);
    return ;
  }
  float expand_ratio;
  math::RandUniform(1,1.f,max_expand_ratio,&expand_ratio,context);
  CAFFE_ENFORCE_GE(expand_ratio,1.0,"max_expand_ratio must be lager than 1.0");
  
  NormalizedBBox expand_bbox;
  ExpandImage(anno_datum.datum(),expand_ratio,&expand_bbox,
    expand_anno_datum->mutable_datum(),param,context);

  expand_anno_datum -> set_type(anno_datum.type());
  

  const bool do_resize = false;
  const bool do_mirror = false;

  TransformAnnotation(anno_datum,do_resize,expand_bbox,do_mirror,
    expand_anno_datum->mutable_annotation_group(),param);

}

// template
// void ExpandImage(const AnnotatedDatum& anno_datum,AnnotatedDatum* expand_anno_datum,
//   TransformParam param,CPUContext* context);

template
void ExpandImage(const AnnotatedDatum& anno_datum,AnnotatedDatum* expand_anno_datum,
  TransformParam param,CUDAContext* context);

template<class Context>
void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
  cv::Mat* crop_img,Context* context)
{
  const int img_height = img.rows;
  const int img_width = img.cols;
  NormalizedBBox clipped_bbox;
  ClipBBox(bbox,&clipped_bbox);
  NormalizedBBox scaled_bbox;
  ScaleBBox(clipped_bbox,img_height,img_width,&scaled_bbox);

  int w_off = static_cast<int>(scaled_bbox.xmin());
  int h_off = static_cast<int>(scaled_bbox.ymin());
  int width = static_cast<int>(scaled_bbox.xmax()-scaled_bbox.xmin());
  int height = static_cast<int>(scaled_bbox.ymax()-scaled_bbox.ymin());

  cv::Rect bbox_roi(w_off,h_off,width,height);
  img(bbox_roi).copyTo(*crop_img);

}

// template
// void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
//   cv::Mat* crop_img,CPUContext* context);

template
void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
  cv::Mat* crop_img,CUDAContext* context);


template<class Context>
void CropImage(const Datum& datum,const NormalizedBBox& bbox,
  Datum* crop_datum,TransformParam param,Context* context)
{
  if(datum.encoded()){
    CAFFE_ENFORCE_EQ( !(param.force_color&&param.force_gray),true,
      "cannot set both force_color and force_gray");
    cv::Mat cv_img;
    if(param.force_color||param.force_gray){
      cv_img = DecodeDatumToCVMat(datum,param.force_color);
    }else{
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    cv::Mat crop_img;
    CropImage(cv_img,bbox,&crop_img,context);
    EncodeCVMatToDatum(crop_img,"jpg",crop_datum);
    crop_datum->set_label(datum.label());
    return ;
  }else{
    if(param.force_color||param.force_gray){
      LOG(ERROR)<<"force_color and force_gray only for encoded datum";
    }
  }
}

// template
// void CropImage(const Datum& datum,const NormalizedBBox& bbox,
//   Datum* crop_datum,TransformParam param,CPUContext* context);

template
void CropImage(const Datum& datum,const NormalizedBBox& bbox,
  Datum* crop_datum,TransformParam param,CUDAContext* context);


template<class Context>
void CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum,
                                       TransformParam param,Context* context) {
  // Crop the datum.
  CropImage(anno_datum.datum(), bbox, cropped_anno_datum->mutable_datum(),param,context);
  cropped_anno_datum->set_type(anno_datum.type());

  // Transform the annotation according to crop_bbox.
  const bool do_resize = false;
  const bool do_mirror = false;
  NormalizedBBox crop_bbox;
  ClipBBox(bbox, &crop_bbox);
  TransformAnnotation(anno_datum, do_resize, crop_bbox, do_mirror,
                      cropped_anno_datum->mutable_annotation_group(),param);
}

// template
// void CropImage(const AnnotatedDatum& anno_datum,
//                                        const NormalizedBBox& bbox,
//                                        AnnotatedDatum* cropped_anno_datum,
//                                        TransformParam param,CPUContext* context);

template
void CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum,
                                       TransformParam param,CUDAContext* context);

}
