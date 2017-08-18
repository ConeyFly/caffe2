#ifndef CAFFE2_IMAGE_ANNOTATION_INPUT_OP_H_
#define CAFFE2_IMAGE_ANNOTATION_INPUT_OP_H_

#include <opencv2/opencv.hpp>

#include <iostream>

#include "caffe/proto/caffe.pb.h"
#include "caffe2/core/db.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/thread_pool.h"
#include "caffe2/operators/prefetch_op.h"
#include "caffe2/image/transform_gpu.h"
#include "caffe2/utils/im_transforms.h"
#include "caffe2/utils/image_util.h"
#include "caffe2/utils/sampler.h"

using namespace std;


namespace caffe2 {


using caffe::AnnotatedDatum;
using caffe::AnnotationGroup ;
using caffe::NormalizedBBox ;
using caffe::Annotation;
using caffe::TransformationParameter;

 


class CUDAContext;

template <class Context>
class AnnotationInputOp final
    : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::context_;
  using PrefetchOperator<Context>::prefetch_thread_;
  explicit AnnotationInputOp(const OperatorDef& operator_def,
                                    Workspace* ws);
  ~AnnotationInputOp() {
    PrefetchOperator<Context>::Finalize();
  }

  bool Prefetch() override;
  bool CopyPrefetched() override;




  // Structure to store per-image information
  // This can be modified by the DecodeAnd* so needs
  // to be privatized per launch.

 private:
  using BoundingBox = struct {
    bool valid;
    int ymin;
    int xmin;
    int height;
    int width;
  };
  using PerImageArg = struct {
    BoundingBox bounding_params;
  };


  bool GetImageAndLabelAndInfoFromDBValue(
      const string& value, cv::Mat* img, PerImageArg& info, int item_id,
      const bool do_mirror);
  void DecodeAndTransform(
      const std::string& value, float *image_data, int item_id,
      const int channels, std::size_t thread_index);
  void DecodeAndTransposeOnly(
      const std::string& value, uint8_t *image_data, int item_id,
      const int channels, std::size_t thread_index);

  unique_ptr<db::DBReader> owned_reader_;
  const db::DBReader* reader_;
  CPUContext cpu_context_;
  TensorCPU prefetched_image_; //Tensor<CPUContext>
  TensorCPU prefetched_label_;
  Tensor<Context> prefetched_image_on_device_;
  Tensor<Context> prefetched_label_on_device_;
  // Default parameters for images
  PerImageArg default_arg_;
  TransformParam transform_param_;
  AnnotatedDataParam annotated_data_param_;


  int batch_size_;
  bool color_;
  int scale_;
  // Minsize is similar to scale except that it will only
  // force the image to scale up if it is too small. In other words,
  // it ensures that both dimensions of the image are at least minsize_
  int minsize_;
  bool warp_;
  int crop_;
  std::vector<float> mean_;
  std::vector<float> std_;
  Tensor<Context> mean_gpu_;
  Tensor<Context> std_gpu_;
  bool mirror_;
  bool is_test_;
  bool use_caffe_datum_;
  bool gpu_transform_;
  bool mean_std_copied_ = false;
  bool anno_type_ ;

  int sum_bboxes;
  int label_idx;

  
  std::map<int, std::vector<AnnotationGroup> > all_anno;
  // thread pool for parse + decode
  int num_decode_threads_;
  std::shared_ptr<TaskThreadPool> thread_pool_;

  // Output type for GPU transform path
  TensorProto_DataType output_type_;

  // Working variables
  std::vector<std::mt19937> randgen_per_thread_;
};

template <class Context>
AnnotationInputOp<Context>::AnnotationInputOp(
      const OperatorDef& operator_def, Workspace* ws)
      : PrefetchOperator<Context>(operator_def, ws),
        reader_(nullptr),
        batch_size_(
            OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
        color_(OperatorBase::template GetSingleArgument<int>("color", 1)),
        scale_(OperatorBase::template GetSingleArgument<int>("scale", -1)),
        minsize_(OperatorBase::template GetSingleArgument<int>("minsize", -1)),
        warp_(OperatorBase::template GetSingleArgument<int>("warp", 0)),
        crop_(OperatorBase::template GetSingleArgument<int>("crop", -1)),
        mirror_(OperatorBase::template GetSingleArgument<int>("mirror", 0)),
        is_test_(OperatorBase::template GetSingleArgument<int>("is_test", 0)),
        anno_type_(OperatorBase::template GetSingleArgument<int>("anno_type",1)),
        use_caffe_datum_(OperatorBase::template GetSingleArgument<int>(
              "use_caffe_datum", 0)),
        gpu_transform_(OperatorBase::template GetSingleArgument<int>(
              "use_gpu_transform", 0)),
        num_decode_threads_(OperatorBase::template GetSingleArgument<int>(
              "decode_threads", 4)),
        thread_pool_(std::make_shared<TaskThreadPool>(num_decode_threads_)),
        // output type only supported with CUDA and use_gpu_transform for now
        output_type_(cast::GetCastDataType(this->arg_helper(), "output_type"))
  {
  /*
  mean_ = OperatorBase::template GetRepeatedArgument<float>(
    "mean_per_channel",
    {OperatorBase::template GetSingleArgument<float>("mean", 0.)});

  std_ = OperatorBase::template GetRepeatedArgument<float>(
    "std_per_channel",
    {OperatorBase::template GetSingleArgument<float>("std", 1.)});
  */
  default_arg_.bounding_params = {
    false,
    OperatorBase::template GetSingleArgument<int>("bounding_ymin", -1),
    OperatorBase::template GetSingleArgument<int>("bounding_xmin", -1),
    OperatorBase::template GetSingleArgument<int>("bounding_height", -1),
    OperatorBase::template GetSingleArgument<int>("bounding_width", -1),
  };

  transform_param_.resize_param = {
    OperatorBase::template GetSingleArgument<bool>("resize_valid",true),
    OperatorBase::template GetSingleArgument<float>("resize_prob",1.0),
    OperatorBase::template GetSingleArgument<string>("resize_mode","WARP"),
    OperatorBase::template GetSingleArgument<int>("resize_height",300),
    OperatorBase::template GetSingleArgument<int>("resize_width",300),
    OperatorBase::template GetSingleArgument<int>("height_scale",0),
    OperatorBase::template GetSingleArgument<int>("width_scale",0),
    OperatorBase::template GetRepeatedArgument<string>("interp_mode"),
  };  



  transform_param_.emit_constraint = {
    OperatorBase::template GetSingleArgument<bool>("emit_valid",true),
    OperatorBase::template GetSingleArgument<string>("emit_type","CENTER"),
  };

  transform_param_.distort_param = {
    OperatorBase::template GetSingleArgument<bool>("distort_valid",true),
    OperatorBase::template GetSingleArgument<float>("brightness_prob",0.5),
    OperatorBase::template GetSingleArgument<float>("brightness_delta",32.0),
    OperatorBase::template GetSingleArgument<float>("contrast_prob",0.5),
    OperatorBase::template GetSingleArgument<float>("contrast_lower",0.5),
    OperatorBase::template GetSingleArgument<float>("contrast_upper",1.5),
    OperatorBase::template GetSingleArgument<float>("hue_prob",0.5),
    OperatorBase::template GetSingleArgument<float>("hue_delat",18.0),
    OperatorBase::template GetSingleArgument<float>("saturation_prob",0.5),
    OperatorBase::template GetSingleArgument<float>("saturation_lower",0.5),
    OperatorBase::template GetSingleArgument<float>("saturation_upper",1.5),
    OperatorBase::template GetSingleArgument<float>("random_order_prob",0.0),
  };

  transform_param_.expand_param = {
    OperatorBase::template GetSingleArgument<bool>("expand_valid",true),
    OperatorBase::template GetSingleArgument<float>("expand_prob",0.5),
    OperatorBase::template GetSingleArgument<float>("max_expand_ratio",4.0),
  };

  
  transform_param_.trans_valid = 
    OperatorBase::template GetSingleArgument<bool>("trans_valid",true);
  transform_param_.mirror = 
    OperatorBase::template GetSingleArgument<bool>("mirror",true);
  transform_param_.crop = 
    OperatorBase::template GetSingleArgument<int>("crop",0);
  transform_param_.mean_value = 
    OperatorBase::template GetRepeatedArgument<float>("mean");
  if(transform_param_.mean_value.size() == 1){
    transform_param_.mean_value.resize(3,transform_param_.mean_value[0]);
  }  
  transform_param_.force_color = 
    OperatorBase::template GetSingleArgument<bool>("force_color",true);
  transform_param_.force_gray =
    OperatorBase::template GetSingleArgument<bool>("force_gray",false);

  BatchSampler tmp_sampler1(1,1,1.0,1.0,1.0,1.0,0.0,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler1);

  BatchSampler tmp_sampler2(1,50,0.3,1.0,0.5,2.0,0.1,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler2);
  
  BatchSampler tmp_sampler3(1,50,0.3,1.0,0.5,2.0,0.3,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler3);

  BatchSampler tmp_sampler4(1,50,0.3,1.0,0.5,2.0,0.5,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler4);

  BatchSampler tmp_sampler5(1,50,0.3,1.0,0.5,2.0,0.7,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler5);

  BatchSampler tmp_sampler6(1,50,0.3,1.0,0.5,2.0,0.9,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler6);

  BatchSampler tmp_sampler7(1,50,0.3,1.0,0.5,2.0,1.0,true);
  annotated_data_param_.batch_sampler.push_back(tmp_sampler7);

  

  // LOG(INFO)<<"resize_prob = "<< transform_param_.resize_param.resize_prob
  // <<"emit_type = "<<transform_param_.emit_constraint.emit_type
  // <<"distort_param = "<<transform_param_.expand_param.max_expand_ratio;

  // annotated_data_param_.batch_sampler = {
  //   OperatorBase::template GetRepeatedArgument<BatchSampler>(
  //     "batch_sampler",
  //   {OperatorBase::template GetSingleArgument<int>("max_sample",1),
  //   OperatorBase::template GetSingleArgument<int>("max_trials",1)})
  // };

  if (operator_def.input_size() == 0) {
    LOG(ERROR) << "You are using an old ImageInputOp format that creates "
                       "a local db reader. Consider moving to the new style "
                       "that takes in a DBReader blob instead.";
    string db_name =
        OperatorBase::template GetSingleArgument<string>("db", "");
    CAFFE_ENFORCE_GT(db_name.size(), 0, "Must specify a db name.");
    owned_reader_.reset(new db::DBReader(
        OperatorBase::template GetSingleArgument<string>(
            "db_type", "leveldb"),
        db_name));
    reader_ = owned_reader_.get();
  }
  /*
  CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be nonnegative.");
  CAFFE_ENFORCE((scale_ > 0) != (minsize_ > 0),
                "Must provide one and only one of scaling or minsize");
  CAFFE_ENFORCE_GT(crop_, 0, "Must provide the cropping value.");
  CAFFE_ENFORCE_GE(
    scale_ > 0 ? scale_ : minsize_,
    crop_, "The scale/minsize value must be no smaller than the crop value.");

  CAFFE_ENFORCE_EQ(
      mean_.size(),
      std_.size(),
      "The mean and std. dev vectors must be of the same size.");
  CAFFE_ENFORCE(mean_.size() == 1 || mean_.size() == 3,
                "The mean and std. dev vectors must be of size 1 or 3");
  */
  if (default_arg_.bounding_params.ymin < 0
      || default_arg_.bounding_params.xmin < 0
      || default_arg_.bounding_params.height < 0
      || default_arg_.bounding_params.width < 0) {
    default_arg_.bounding_params.valid = false;
  } else {
    default_arg_.bounding_params.valid = true;
  }

  // if (mean_.size() == 1) {
  //   // We are going to extend to 3 using the first value
  //   mean_.resize(3, mean_[0]);
  //   std_.resize(3, std_[0]);
  // }

//  LOG(INFO)<<"resize_param.prob = "<<transform_param_.resize_param.prob
//  <<", expand_param.prob = "<<transform_param_.expand_param.prob;

  LOG(INFO) << "Creating an image input op with the following setting: ";
  LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
  if (gpu_transform_) {
    LOG(INFO) << "    Performing transformation on GPU";
  }
  LOG(INFO) << "    Outputting in batches of " << batch_size_ << " images;";
  LOG(INFO) << "    Treating input image as "
            << (color_ ? "color " : "grayscale ") << "image;";
  if (default_arg_.bounding_params.valid) {
    LOG(INFO) << "    Applying a default bounding box of Y ["
              << default_arg_.bounding_params.ymin << "; "
              << default_arg_.bounding_params.ymin +
      default_arg_.bounding_params.height
              << ") x X ["
              << default_arg_.bounding_params.xmin << "; "
              << default_arg_.bounding_params.xmin +
      default_arg_.bounding_params.width
              << ")";
  }
  if (scale_ > 0) {
    LOG(INFO) << "    Scaling image to " << scale_
              << (warp_ ? " with " : " without ") << "warping;";
  } else {
    // Here, minsize_ > 0
    LOG(INFO) << "    Ensuring minimum image size of " << minsize_
              << (warp_ ? " with " : " without ") << "warping;";
  }
  LOG(INFO) << "    " << (is_test_ ? "Central" : "Random")
            << " cropping image to " << crop_
            << (mirror_ ? " with " : " without ") << "random mirroring;";

  auto mit = mean_.begin();
  auto sit = std_.begin();

  for (int i = 0;
       mit != mean_.end() && sit != std_.end();
       ++mit, ++sit, ++i) {
    LOG(INFO) << "    Default [Channel " << i << "] Subtract mean " << *mit
              << " and divide by std " << *sit << ".";
    // We actually will use the inverse of std, so inverse it here
    *sit = 1.f / *sit;
  }
  LOG(INFO) << "    Outputting images as "
            << OperatorBase::template GetSingleArgument<string>("output_type", "unknown") << ".";

  std::mt19937 meta_randgen(time(nullptr));
  for (int i = 0; i < num_decode_threads_; ++i) {
    randgen_per_thread_.emplace_back(meta_randgen());
  }
  prefetched_image_.Resize(
      TIndex(batch_size_),
      TIndex(color_ ? 3 : 1),
      TIndex(transform_param_.resize_param.height),
      TIndex(transform_param_.resize_param.width));
  prefetched_label_.Resize(TIndex(1), TIndex(1), TIndex(20),TIndex(8));
  
 // LOG(INFO)<<"height = "<< transform_param_.resize_param.height;
 // LOG(INFO)<<"width = "<<transform_param_.resize_param.width;
    


}

template <class Context>
bool AnnotationInputOp<Context>::GetImageAndLabelAndInfoFromDBValue(
    const string& value,
    cv::Mat* img,
    PerImageArg& info,
    int item_id,
    const bool do_mirror) {
  //
  // recommend using --caffe2_use_fatal_for_enforce=1 when using ImageInputOp
  // as this function runs on a worker thread and the exceptions from
  // CAFFE_ENFORCE are silently dropped by the thread worker functions
  //
  cv::Mat src;
  info = default_arg_;
  if (use_caffe_datum_) {
    // The input is a caffe datum format.
    AnnotatedDatum anno_datum;
    CAFFE_ENFORCE(anno_datum.ParseFromString(value));


    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum; 

    if(transform_param_.distort_param.distort_valid){
      distort_datum.CopyFrom(anno_datum);
      DistortImage(anno_datum.datum(),distort_datum.mutable_datum(),transform_param_,&context_);
      if(transform_param_.expand_param.expand_valid){
        expand_datum = new AnnotatedDatum();
        ExpandImage(distort_datum,expand_datum,transform_param_,&context_);
      }else{
        expand_datum = &distort_datum;
      }

    }else{
      if(transform_param_.expand_param.expand_valid){
        expand_datum = new AnnotatedDatum();
        ExpandImage(anno_datum,expand_datum,transform_param_,&context_);
      }else{
        expand_datum = &anno_datum;
      }
    }


    AnnotatedDatum* sampled_datum = NULL;
    bool has_sample = false;

    if(annotated_data_param_.batch_sampler.size()>0){
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum,annotated_data_param_.batch_sampler,
        &sampled_bboxes,&context_);
      if(sampled_bboxes.size()>0){
          int rand_idx = math::randomNumberSeed()%sampled_bboxes.size();
          sampled_datum = new AnnotatedDatum();
          CropImage(*expand_datum,sampled_bboxes[rand_idx],sampled_datum,transform_param_,
            &context_);
          has_sample = true;
      }else{
        sampled_datum = expand_datum;
      }
    }else{
      sampled_datum = expand_datum;
    }

    anno_datum = *expand_datum;
    vector<AnnotationGroup> transformed_anno_vec;
    const bool do_resize = transform_param_.resize_param.resize_valid; 

    NormalizedBBox crop_bbox;
    crop_bbox.set_xmin(0.);
    crop_bbox.set_ymin(0.);
    crop_bbox.set_xmax(1.);
    crop_bbox.set_ymax(1.);

    TransformAnnotation(anno_datum,do_resize,crop_bbox,do_mirror,&transformed_anno_vec,
    transform_param_,&context_);

    vector<AnnotationGroup> anno_vec;
    for(int g = 0;g < transformed_anno_vec.size(); ++g) {
      sum_bboxes += transformed_anno_vec[g].annotation_size();
    }


    all_anno[item_id]=transformed_anno_vec;


    //prefetch label 
    if (anno_datum.datum().encoded()) {
      // encoded image in datum.
      src = cv::imdecode(
          cv::Mat(
              1,
              anno_datum.datum().data().size(),
              CV_8UC1,
              const_cast<char*>(anno_datum.datum().data().data())),
          color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    } else {
      // Raw image in datum.
      CAFFE_ENFORCE(anno_datum.datum().channels() == 3 || anno_datum.datum().channels() == 1);

      int src_c = anno_datum.datum().channels();
      src.create(
          anno_datum.datum().height(), anno_datum.datum().width(), (src_c == 3) ? CV_8UC3 : CV_8UC1);

      if (src_c == 1) {
        memcpy(src.ptr<uchar>(0), anno_datum.datum().data().data(), anno_datum.datum().data().size());
      } else {
        // Datum stores things in CHW order, let's do HWC for images to make
        // things more consistent with conventional image storage.
        for (int c = 0; c < 3; ++c) {
          const char* datum_buffer =
              anno_datum.datum().data().data() + anno_datum.datum().height() * anno_datum.datum().width() * c;
          uchar* ptr = src.ptr<uchar>(0) + c;
          for (int h = 0; h < anno_datum.datum().height(); ++h) {
            for (int w = 0; w < anno_datum.datum().width(); ++w) {
              *ptr = *(datum_buffer++);
              ptr += 3;
            }
          }
        }
      }
    }
  } else {
    // The input is a caffe2 format.
    TensorProtos protos;
    CAFFE_ENFORCE(protos.ParseFromString(value));
    const TensorProto& image_proto = protos.protos(0);
    const TensorProto& label_proto = protos.protos(1);
    if (protos.protos_size() == 3) {
      // We have bounding box information
      const TensorProto& bounding_proto = protos.protos(2);
      DCHECK_EQ(bounding_proto.data_type(), TensorProto::INT32);
      DCHECK_EQ(bounding_proto.int32_data_size(), 4);
      info.bounding_params.valid = true;
      info.bounding_params.ymin = bounding_proto.int32_data(0);
      info.bounding_params.xmin = bounding_proto.int32_data(1);
      info.bounding_params.height = bounding_proto.int32_data(2);
      info.bounding_params.width = bounding_proto.int32_data(3);
    }

    if (image_proto.data_type() == TensorProto::STRING) {
      // encoded image string.
      DCHECK_EQ(image_proto.string_data_size(), 1);
      const string& encoded_image_str = image_proto.string_data(0);
      int encoded_size = encoded_image_str.size();
      // We use a cv::Mat to wrap the encoded str so we do not need a copy.
      src = cv::imdecode(
          cv::Mat(
              1,
              &encoded_size,
              CV_8UC1,
              const_cast<char*>(encoded_image_str.data())),
          color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    } else if (image_proto.data_type() == TensorProto::BYTE) {
      // raw image content.
      int src_c = (image_proto.dims_size() == 3) ? image_proto.dims(2) : 1;
      CAFFE_ENFORCE(src_c == 3 || src_c == 1);

      src.create(
          image_proto.dims(0),
          image_proto.dims(1),
          (src_c == 3) ? CV_8UC3 : CV_8UC1);
      memcpy(
          src.ptr<uchar>(0),
          image_proto.byte_data().data(),
          image_proto.byte_data().size());
    } else {
      LOG(FATAL) << "Unknown image data type.";
    }

    if (label_proto.data_type() == TensorProto::FLOAT) {
      DCHECK_EQ(label_proto.float_data_size(), 1);

      prefetched_label_.mutable_data<float>()[item_id] =
          label_proto.float_data(0);
    } else if (label_proto.data_type() == TensorProto::INT32) {
      DCHECK_EQ(label_proto.int32_data_size(), 1);

      prefetched_label_.mutable_data<int>()[item_id] =
          label_proto.int32_data(0);
    } else {
      LOG(FATAL) << "Unsupported label type.";
    }
  }

  //
  // convert source to the color format requested from Op
  //
  int out_c = color_ ? 3 : 1;
  if (out_c == src.channels()) {
    *img = src;
  } else {
    cv::cvtColor(src, *img, (out_c == 1) ? CV_BGR2GRAY : CV_GRAY2BGR);
  }

  // Note(Yangqing): I believe that the mat should be created continuous.
  CAFFE_ENFORCE(img->isContinuous());

  // Sanity check now that we decoded everything

  // Ensure that the bounding box is legit
  if (info.bounding_params.valid
      && (src.rows < info.bounding_params.ymin + info.bounding_params.height
        || src.cols < info.bounding_params.xmin + info.bounding_params.width
     )) {
    info.bounding_params.valid = false;
  }

  // Apply the bounding box if requested
  if (info.bounding_params.valid) {
    // If we reach here, we know the parameters are sane
    cv::Rect bounding_box(info.bounding_params.xmin, info.bounding_params.ymin,
                          info.bounding_params.width, info.bounding_params.height);
    *img = (*img)(bounding_box);

  } else {
    // LOG(INFO) << "No bounding\n";
  }

  
  cv::Mat resize_img;
  resize_img = ApplyResize(*img,transform_param_.resize_param,&context_);
/*  *img = resize_img;
    cv::resize(
        *img,
        resize_img,
        cv::Size(resize_width_, resize_height_),
        0,
        0,
        cv::INTER_AREA);*/
  *img = resize_img;

  
  return true;
}


// Factored out image transformation
template <class Context>
void TransformImage(
    const cv::Mat& img,
    const int channels,
    float* image_data,
    const TransformParam param,
    const int item_id,
    Context* context,
    const bool do_mirror,
    bool is_test = false) {
  
  // const int channels = img.channels();
  const int height = img.rows;
  const int width = img.cols;
  const int crop = param.crop;
  const bool mirror = param.mirror;
  const std::vector<float>mean = param.mean_value;

  CAFFE_ENFORCE_GE(
      height, crop, "Image height must be bigger than crop.");
  CAFFE_ENFORCE_GE(
      width, crop, "Image width must be bigger than crop.");

  const bool do_resize = param.resize_param.resize_valid;

  int img_height = height;
  int img_width = width;

  CAFFE_ENFORCE_GE(img_height,crop);
  CAFFE_ENFORCE_GE(img_width,crop);

  
  int h_off = 0;
  int w_off = 0;


  float img_element;
  int top_index;
  
  
  for (int h = 0; h < height; ++h) {
    int h_idx = h ;
    for (int w = 0; w < width ; ++w) {
      int w_idx = w;
      if(do_mirror){
        w_idx = (width - 1 - w);
      }
      int h_idx_real = h_idx;
      int w_idx_real = w_idx;
      const uint8_t* cv_data = img.ptr(h) + w*channels;
      for (int c = 0; c < channels; ++c) {
        int top_index = (c * height + h_idx_real) * width + w_idx_real;
        image_data[top_index] = static_cast<float>(cv_data[c]);
      }
    }
  }
  

}

// Only crop / transose the image
// leave in uint8_t dataType
template <class Context>
void CropTransposeImage(const cv::Mat& scaled_img, const int channels,
                        uint8_t *cropped_data, const int crop,
                        const bool mirror, std::mt19937 *randgen,
                        std::bernoulli_distribution *mirror_this_image,
                        bool is_test = false) {
  CAFFE_ENFORCE_GE(
      scaled_img.rows, crop, "Image height must be bigger than crop.");
  CAFFE_ENFORCE_GE(
      scaled_img.cols, crop, "Image width must be bigger than crop.");

  // find the cropped region, and copy it to the destination matrix with
  // mean subtraction and scaling.
  int width_offset, height_offset;
  if (is_test) {
    width_offset = (scaled_img.cols - crop) / 2;
    height_offset = (scaled_img.rows - crop) / 2;
  } else {
    width_offset =
      std::uniform_int_distribution<>(0, scaled_img.cols - crop)(*randgen);
    height_offset =
      std::uniform_int_distribution<>(0, scaled_img.rows - crop)(*randgen);
  }

  int idx = 0;
  if (mirror && (*mirror_this_image)(*randgen)) {
    // Copy mirrored image.
  
    for (int h = height_offset; h < height_offset + crop; ++h) {
      for (int w = width_offset + crop - 1; w >= width_offset; --w) {
        const uint8_t* cv_data = scaled_img.ptr(h) + w*channels;
        for (int c = 0; c < channels; ++c) {
          cropped_data[idx++] = cv_data[c];
        }
      }
    }
    
  } else {
    // Copy normally.
    height_offset = width_offset = 0;
    for (int h = height_offset; h < height_offset + crop; ++h) {
      int h_idx = h ;
      for (int w = width_offset; w < width_offset + crop; ++w) {
        int w_idx = w;
        int h_idx_real = h_idx;
        int w_idx_real = w_idx;
        const uint8_t* cv_data = scaled_img.ptr(h) + w*channels;
        for (int c = 0; c < channels; ++c) {
          int top_index = (c*crop + h_idx_real)*crop+w_idx_real;
          cropped_data[top_index] = cv_data[c];
        }
      }
    }
  }
}

// Parse datum, decode image, perform transform
// Intended as entry point for binding to thread pool
template <class Context>
void AnnotationInputOp<Context>::DecodeAndTransform(
      const std::string& value, float *image_data, int item_id,
      const int channels, std::size_t thread_index) {

  CAFFE_ENFORCE((int)thread_index < num_decode_threads_);

  std::bernoulli_distribution mirror_this_image(0.5f);
  std::mt19937* randgen = &(randgen_per_thread_[thread_index]);
  const bool do_mirror = transform_param_.mirror &&(mirror_this_image)(*randgen);
  cv::Mat img;
  // Decode the image
  PerImageArg info;
  CHECK(GetImageAndLabelAndInfoFromDBValue(value, &img, info, item_id,do_mirror));

  // Factor out the image transformation
  TransformImage<Context>(img, channels, image_data, transform_param_,item_id,
                         &context_,do_mirror,is_test_);
}

template <class Context>
void AnnotationInputOp<Context>::DecodeAndTransposeOnly(
    const std::string& value, uint8_t *image_data, int item_id,
    const int channels, std::size_t thread_index) {

  CAFFE_ENFORCE((int)thread_index < num_decode_threads_);

  std::bernoulli_distribution mirror_this_image(0.5f);
  std::mt19937* randgen = &(randgen_per_thread_[thread_index]);
  const bool do_mirror = transform_param_.mirror &&(mirror_this_image)(*randgen);
  cv::Mat img;
  // Decode the image
  PerImageArg info;
//  CHECK(GetImageAndLabelAndInfoFromDBValue(value, &img, info, item_id,
//    randgen,&mirror_this_image));
  CHECK(GetImageAndLabelAndInfoFromDBValue(value, &img, info, item_id,do_mirror));
  // Factor out the image transformation
  CropTransposeImage<Context>(img, channels, image_data, crop_, mirror_,
                              randgen, &mirror_this_image, is_test_);
}


template <class Context>
bool AnnotationInputOp<Context>::Prefetch() {
  if (!owned_reader_.get()) {   
    // if we are not owning the reader, we will get the reader pointer from
    // input. Otherwise the constructor should have already set the reader
    // pointer.
    reader_ = &OperatorBase::Input<db::DBReader>(0);
  }
  const int channels = color_ ? 3 : 1;
  // Call mutable_data() once to allocate the underlying memory.
  if (gpu_transform_) {
    // we'll transfer up in int8, then convert later
    prefetched_image_.mutable_data<uint8_t>();
  } else {
    prefetched_image_.mutable_data<float>();
  }
  prefetched_label_.mutable_data<float>();
  // Prefetching handled with a thread pool of "decode_threads" threads.

  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    std::string key, value;
    cv::Mat img;
    
    // read data
    reader_->Read(&key, &value);

    // determine label type based on first item
    if( item_id == 0 ) {
      
      label_idx = 0;
      sum_bboxes = 0;
      if( use_caffe_datum_ ) {

        prefetched_label_.mutable_data<float>();


      } else {
        TensorProtos protos;
        CAFFE_ENFORCE(protos.ParseFromString(value));
        TensorProto_DataType labeldt = protos.protos(1).data_type();
        if( labeldt == TensorProto::INT32 ) {
          prefetched_label_.mutable_data<int>();
        } else if ( labeldt == TensorProto::FLOAT) {
          prefetched_label_.mutable_data<float>();
        } else {
          LOG(FATAL) << "Unsupported label type.";
        }
      }
    }

    // launch into thread pool for processing
    if (gpu_transform_) {
      // output of decode will still be int8
      uint8_t* image_data = prefetched_image_.mutable_data<uint8_t>() +
          crop_ * crop_ * channels * item_id;
      thread_pool_->runTaskWithID(std::bind(
          &AnnotationInputOp<Context>::DecodeAndTransposeOnly,
          this,
          std::string(value),
          image_data,
          item_id,
          channels,
          std::placeholders::_1));
    } else {
      float* image_data = prefetched_image_.mutable_data<float>() +
          crop_ * crop_ * channels * item_id;
    //  LOG(INFO)<<"data_size = "<< crop_ * crop_ * channels ;
      thread_pool_->runTaskWithID(std::bind(
          &AnnotationInputOp<Context>::DecodeAndTransform,
          this,
          std::string(value),
          image_data,
          item_id,
          channels,
          std::placeholders::_1));
    }
  }
  thread_pool_->waitWorkComplete();
  prefetched_label_.Resize(TIndex(1),TIndex(1),TIndex(sum_bboxes),TIndex(8));
  
 // float* label_index = prefetched_label_.mutable_data<float>();
 // LOG(INFO)<<"label_index ok";
  int idx = 0;
  for(int item_id = 0 ; item_id < batch_size_ ; ++item_id){
    const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
    for(int g = 0; g < anno_vec.size(); ++g){
      const AnnotationGroup& anno_group = anno_vec[g];
      for(int a = 0; a < anno_group.annotation_size() ; a++){
        const Annotation& anno = anno_group.annotation(a);
        const NormalizedBBox& bbox = anno.bbox();
        prefetched_label_.mutable_data<float>()[idx++] = item_id;
        prefetched_label_.mutable_data<float>()[idx++] = anno_group.group_label();
        prefetched_label_.mutable_data<float>()[idx++] = anno.instance_id();
        prefetched_label_.mutable_data<float>()[idx++] = bbox.xmin();
        prefetched_label_.mutable_data<float>()[idx++] = bbox.ymin();
        prefetched_label_.mutable_data<float>()[idx++] = bbox.xmax();
        prefetched_label_.mutable_data<float>()[idx++] = bbox.ymax();
        prefetched_label_.mutable_data<float>()[idx++] = bbox.difficult(); 
      }
    }
  }
  all_anno.clear();
  // If the context is not CPUContext, we will need to do a copy in the
  // prefetch function as well.
  if (!std::is_same<Context, CPUContext>::value) {
    // prefetched_image_on_device_.Resize(TIndex(batch_size_),
    //   TIndex(color_ ? 3 : 1),
    //   TIndex(transform_param_.resize_param.height),
    //   TIndex(transform_param_.resize_param.width));
    // prefetched_label_on_device_.Resize(TIndex(1),TIndex(1),TIndex(sum_bboxes),TIndex(8));
    prefetched_image_on_device_.CopyFrom(prefetched_image_, &context_);
    prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
    // CHECK(context_.FinishDeviceComputation());
  }
  return true;
}

template <class Context>
bool AnnotationInputOp<Context>::CopyPrefetched() {
  
  auto* image_output = OperatorBase::Output<Tensor<Context> >(0);
  auto* label_output = OperatorBase::Output<Tensor<Context> >(1);
  // Note(jiayq): The if statement below should be optimized away by the
  // compiler since std::is_same is a constexpr.
  
  if (std::is_same<Context, CPUContext>::value) {
    image_output->CopyFrom(prefetched_image_, &context_);
    label_output->CopyFrom(prefetched_label_, &context_);
  } else {
    
    if (gpu_transform_) {
      if (!mean_std_copied_) {
        mean_gpu_.Resize(mean_.size());
        std_gpu_.Resize(std_.size());

        context_.template Copy<float, CPUContext, Context>(
          mean_.size(), mean_.data(), mean_gpu_.template mutable_data<float>());
        context_.template Copy<float, CPUContext, Context>(
          std_.size(), std_.data(), std_gpu_.template mutable_data<float>());
        mean_std_copied_ = true;
      }
      // GPU transform kernel allows explicitly setting output type
      if (output_type_ == TensorProto_DataType_FLOAT) {
        TransformOnGPU<uint8_t,float,Context>(prefetched_image_on_device_,
                                              image_output, mean_gpu_,
                                              std_gpu_, &context_);
      } else if (output_type_ == TensorProto_DataType_FLOAT16) {
        TransformOnGPU<uint8_t,float16,Context>(prefetched_image_on_device_,
                                                image_output, mean_gpu_,
                                                std_gpu_, &context_);
      }  else {
        return false;
      }
    } else {
      
      image_output->CopyFrom(prefetched_image_on_device_, &context_);
    }
    
    label_output->CopyFrom(prefetched_label_on_device_, &context_);
  }
  return true;
}
}  // namespace caffe2

#endif  // CAFFE2_IMAGE_IMAGE_INPUT_OP_H_
