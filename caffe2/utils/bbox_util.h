#ifndef CAFFE2_UTILS_BBOX_UTIL_H_
#define CAFFE2_UTILS_BBOX_UTIL_H_

#include <vector>
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <float.h>
#include "caffe/proto/caffe.pb.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/image_util.h"
#include "caffe2/core/tensor.h"
//#include "caffe2/utils/im_transforms.h"


namespace caffe2 {
using std::map;
using std::pair;
using caffe::NormalizedBBox;
using caffe::PriorBoxParameter_CodeType;
using caffe::MultiBoxLossParameter;
using caffe::MultiBoxLossParameter_MatchType;
using caffe::MultiBoxLossParameter_ConfLossType;
using caffe::MultiBoxLossParameter_LocLossType;
using caffe::MultiBoxLossParameter_MiningType;
//using caffe2::EmitConstraint;

typedef PriorBoxParameter_CodeType CodeType;
typedef map<int, vector<NormalizedBBox> > LabelBBox;
typedef MultiBoxLossParameter_MatchType MatchType;
typedef MultiBoxLossParameter_ConfLossType ConfLossType;
typedef MultiBoxLossParameter_LocLossType LocLossType;
typedef MultiBoxLossParameter_MiningType MiningType;

bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);
  

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);


void MineHardExamples(TensorCPU conf_tensor,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);


void EncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
  const map<int,vector<NormalizedBBox> >&all_gt_bboxes,
  const vector<map<int,vector<int> > >&all_match_indices,
  const vector<NormalizedBBox>& prior_bboxes,
  const vector<vector<float> >& prior_variances,
  const MultiBoxLossParameter& multibox_loss_param,
  float* loc_pred_data,float* loc_gt_data);

void EncodeConfPrediction(const float* conf_data,const int num,
  const int num_priors,const MultiBoxLossParameter& multibox_loss_param,
  const vector<map<int,vector<int> > >&all_match_indices,
  const vector<vector<int> >& all_neg_indices,
  const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
  float* conf_pred_data,float* conf_gt_data);

void ComputeConfLoss(const float* conf_data,const int num,
  const int num_preds_per_class,const int num_classes,
  const int background_label_id,const ConfLossType loss_type,
  vector<vector<float> >* all_conf_loss);

void ComputeConfLoss(const float* conf_data,const int num,
  const int num_preds_per_class,const int num_classes,
  const int background_label_id,const ConfLossType loss_type,
  const vector<map<int,vector<int> > >& all_match_indices,
  const map<int ,vector<NormalizedBBox> >& all_gt_bboxes,
  vector<vector<float> >* all_conf_loss);



template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data,const int num_gt,
  const int background_label_id,const bool use_difficult_gt,
  map<int, vector<NormalizedBBox> >* all_gt_bboxes);

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_scores);

template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);

void FindMatches(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      vector<map<int, vector<float> > >* all_match_overlaps,
      vector<map<int, vector<int> > >* all_match_indices);

void MatchBBox(const vector<NormalizedBBox>& gt_bboxes,
    const vector<NormalizedBBox>& pred_bboxes, const int label,
    const MatchType match_type, const float overlap_threshold,
    const bool ignore_cross_boundary_bbox,
    vector<int>* match_indices, vector<float>* match_overlaps);

// Compute bbox size.
float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);

// Decode all bboxes in a batch.
void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_pred,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, vector<LabelBBox>* all_decode_bboxes);
// Decode a set of bboxes according to a set of prior bboxes.
void DecodeBBoxes(const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes);
// Decode a bbox according to a prior bbox.
void DecodeBBox(const NormalizedBBox& prior_bbox,
    const vector<float>& prior_variance, const CodeType code_type,
    const bool variance_encoded_in_target, const bool clip_bbox,
    const NormalizedBBox& bbox, NormalizedBBox* decode_bbox);

void EncodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const CodeType code_type, const bool encode_variance_in_target,
    const NormalizedBBox& bbox, NormalizedBBox* encode_bbox) ;

// Clip the NormalizedBBox such that the range for each corner is [0, 1].
void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox);

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices);
void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2);
// Compute the jaccard (intersection over union IoU) overlap between two bboxes.
float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized = true);
// Compute the intersection between two bboxes.
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox);

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);

template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec);

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2);

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized = true);

float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2);

void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                NormalizedBBox* loc_bbox);

bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
                        const NormalizedBBox& bbox,
                        const EmitConstraintParam& emit_constraint);

bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox);

void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox);

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections);

void OutputBBox(const NormalizedBBox& bbox, const pair<int, int>& img_size,
                const bool has_resize, const ResizeParam& resize_param,
                NormalizedBBox* out_bbox);

} // namspace caffe2

#endif
