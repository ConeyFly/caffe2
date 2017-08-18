#ifndef CAFFE2_UTILS_SAMPLER_H_
#define CAFFE2_UTILS_SAMPLER_H_

#include <vector>

#include "caffe2/core/logging.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe2/utils/bbox_util.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {


using caffe::NormalizedBBox;
using caffe::AnnotatedDatum;
using caffe::SampleConstraint;
//using caffe::BatchSampler;
using caffe::Sampler;
using caffe::AnnotationGroup;
using caffe::Annotation;

//template <class Context>
//using Operator<Context>::context_;

// Find all annotated NormalizedBBox.
void GroupObjectBBoxes(const AnnotatedDatum& anno_datum,
                       vector<NormalizedBBox>* object_bboxes);

// Check if a sampled bbox satisfy the constraints with all object bboxes.
bool SatisfySampleConstraint(const NormalizedBBox& sampled_bbox,
                             const vector<NormalizedBBox>& object_bboxes,
                             const BatchSampler& sample_constraint);

// Sample a NormalizedBBox given the specifictions.
template <class Context>
void SampleBBox(const BatchSampler& sampler, NormalizedBBox* sampled_bbox,
	Context* context);

// Generate samples from NormalizedBBox using the BatchSampler.
template<class Context>
void GenerateSamples(const NormalizedBBox& source_bbox,
                     const vector<NormalizedBBox>& object_bboxes,
                     const BatchSampler& batch_sampler,
                     vector<NormalizedBBox>* sampled_bboxes,
                     Context* context);

// Generate samples from AnnotatedDatum using the BatchSampler.
// All sampled bboxes which satisfy the constraints defined in BatchSampler
// is stored in sampled_bboxes.
template<class Context>
void GenerateBatchSamples(const AnnotatedDatum& anno_datum,
                          const vector<BatchSampler>& batch_samplers,
                          vector<NormalizedBBox>* sampled_bboxes,
                          Context* context);

}  // namespace caffe2

#endif  // CAFFE_UTIL_SAMPLER_H_
