#ifndef CAFFE2_UTILS_IM_TRANSFORMS_H_
#define CAFFE2_UTILS_IM_TRANSFORMS_H_


#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include <opencv2/opencv.hpp>
#include "caffe2/utils/bbox_util.h"
#include "caffe/proto/caffe.pb.h"
//#include "caffe2/utils/image_util.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2{

using caffe::Datum;
using caffe::AnnotatedDatum;
using caffe::AnnotationGroup ;
using caffe::NormalizedBBox ;
using caffe::Annotation;
using caffe::TransformationParameter;
using caffe::ResizeParameter;
//using ::google::protobuf::RepeatedPtrField;

const float prob_eps = 0.01;

template<class Context>
void TransformAnnotation(const AnnotatedDatum& anno_datum,const bool do_resize,
  const NormalizedBBox& crop_bbox,const bool do_mirror,
  vector<AnnotationGroup>* transformed_anno_vec,TransformParam param,Context* context);

template <class Context>
cv::Mat ApplyDistort(const cv::Mat& in_img,
	const DistortParam& param,Context* context);

template <class Context>
cv::Mat ApplyResize(const cv::Mat& in_img,
	const ResizeParam& param,Context* context);



template <class Context>
int roll_weighted_die(const std::vector<float>& probabilities,Context* context);


template <class Context>
void DistortImage(const Datum& datum,Datum* distort_datum,TransformParam param,Context* context);

template <class Context>
void ExpandImage(const cv::Mat& img,const float expand_ratio,
	NormalizedBBox* expand_bbox,cv::Mat* expand_img,TransformParam param,Context* context);

template <class Context>
void ExpandImage(const Datum& datum,const float expand_ratio,NormalizedBBox* expand_bbox,
	Datum* expand_datum,TransformParam param,Context* context);

template <class Context>
void ExpandImage(const AnnotatedDatum& anno_datum,AnnotatedDatum* expand_anno_datum,
	TransformParam param,Context* context);

void UpdateBBoxByResizePolicy(const ResizeParam& param,const int old_width,
  const int old_height,NormalizedBBox* bbox);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);

cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void EncodeCVMatToDatum(const cv::Mat& cv_img, const string& encoding,
                        Datum* datum);
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) ;
bool DecodeDatum(Datum* datum, bool is_color) ;

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);
void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img);
void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);
void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

template<class Context>
void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta,Context* context);

template<class Context>
void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper,Context* context);

template<class Context>
void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper,Context* context);

template<class Context>
void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta,Context* context);

template<class Context>
void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob,Context* context);

template<class Context>
void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
  cv::Mat* crop_img,Context* context);

template<class Context>
void CropImage(const Datum& datum,const NormalizedBBox& bbox,
  Datum* crop_datum,TransformParam param,Context* context);

template<class Context>
void CropImage(const AnnotatedDatum& anno_datum,
                                       const NormalizedBBox& bbox,
                                       AnnotatedDatum* cropped_anno_datum,
                                       TransformParam param,Context* context);

}

#endif