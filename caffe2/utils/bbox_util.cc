#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe2/core/tensor.h"
#include "caffe2/utils/bbox_util.h"

namespace caffe2 {

bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  return bbox1.score() < bbox2.score();
}

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  return bbox1.score() > bbox2.score();
}

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data,const int num_gt,
  const int background_label_id,const bool use_difficult_gt,
  map<int, vector<NormalizedBBox> >* all_gt_bboxes){
  all_gt_bboxes->clear();

  for(int i=0;i < num_gt ; ++i){
    int start_idx = i * 8;
    int item_id = gt_data[start_idx];
    if(item_id == -1){
      continue;
    }
    int label = gt_data[start_idx + 1];
   
    CAFFE_ENFORCE_NE(background_label_id,label,
      "Found background label in the dataset.");
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if(!use_difficult_gt && difficult){
      continue;
    }
    NormalizedBBox bbox;
    bbox.set_label(label);
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_gt_bboxes)[item_id].push_back(bbox);
  }
}

template void GetGroundTruth(const float* gt_data,const int num_gt,
  const int background_label_id,const bool use_difficult_gt,
  map<int,vector<NormalizedBBox> >* all_gt_bboxes);

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes) {
  all_gt_bboxes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 8;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      break;
    }
    NormalizedBBox bbox;
    int label = gt_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);
template void GetGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);


template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds) {
  loc_preds->clear();
  if (share_location) {
    CAFFE_ENFORCE_EQ(num_loc_classes, 1);
  }
  loc_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    LabelBBox& label_bbox = (*loc_preds)[i]; // the i sample
    for (int p = 0; p < num_preds_per_class; ++p) { // num_preds_per_class = 8732
      int start_idx = p * num_loc_classes * 4; // num_loc_classes = 1
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c; // label = -1
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
        label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
        label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
        label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}
// Explicit initialization.
template void GetLocPredictions(const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float> >& label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].push_back(conf_data[start_idx + c]);
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}
// Explicit initialization.
template void GetConfidenceScores(const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_preds);

template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

bool IsCrossBoundaryBBox(const NormalizedBBox& bbox) {
  return bbox.xmin() < 0 || bbox.xmin() > 1 ||
      bbox.ymin() < 0 || bbox.ymin() > 1 ||
      bbox.xmax() < 0 || bbox.xmax() > 1 ||
      bbox.ymax() < 0 || bbox.ymax() > 1;
}

void MatchBBox(const vector<NormalizedBBox>& gt_bboxes,
    const vector<NormalizedBBox>& pred_bboxes, const int label,
    const MatchType match_type, const float overlap_threshold,
    const bool ignore_cross_boundary_bbox,
    vector<int>* match_indices, vector<float>* match_overlaps) {
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  match_indices->resize(num_pred, -1);
  match_overlaps->clear();
  match_overlaps->resize(num_pred, 0.);

  int num_gt = 0;
  vector<int> gt_indices;
  if (label == -1) {
    // label -1 means comparing against all ground truth.
    num_gt = gt_bboxes.size();
    for (int i = 0; i < num_gt; ++i) {
      gt_indices.push_back(i);
    }
  } else {
    // Count number of ground truth boxes which has the desired label.
    for (int i = 0; i < gt_bboxes.size(); ++i) {
      if (gt_bboxes[i].label() == label) {
        num_gt++;
        gt_indices.push_back(i);
      }
    }
  }
  if (num_gt == 0) {
    return;
  }

  // Store the positive overlap between predictions and ground truth.
  map<int, map<int, float> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    if (ignore_cross_boundary_bbox && IsCrossBoundaryBBox(pred_bboxes[i])) {
      (*match_indices)[i] = -2;
      continue;
    }
    for (int j = 0; j < num_gt; ++j) {
      float overlap = JaccardOverlap(pred_bboxes[i], gt_bboxes[gt_indices[j]]);
      if (overlap > 1e-6) {
        (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
        overlaps[i][j] = overlap;
      }
    }
  }

  // Bipartite matching.
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if ((*match_indices)[i] != -1) {
        // The prediction already has matched ground truth or is ignored.
        continue;
      }
      for (int p = 0; p < gt_pool.size(); ++p) {
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          // No overlap between the i-th prediction and j-th ground truth.
          continue;
        }
        // Find the maximum overlapped pair.
        if (it->second[j] > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ((*match_indices)[max_idx], -1);
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      // Erase the ground truth.
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  switch (match_type) {
    case caffe::MultiBoxLossParameter_MatchType_BIPARTITE:
      // Already done.
      break;
    case caffe::MultiBoxLossParameter_MatchType_PER_PREDICTION:
      // Get most overlaped for the rest prediction bboxes.
      for (map<int, map<int, float> >::iterator it = overlaps.begin();
           it != overlaps.end(); ++it) {
        int i = it->first;
        if ((*match_indices)[i] != -1) {
          // The prediction already has matched ground truth or is ignored.
          continue;
        }
        int max_gt_idx = -1;
        float max_overlap = -1;
        for (int j = 0; j < num_gt; ++j) {
          if (it->second.find(j) == it->second.end()) {
            // No overlap between the i-th prediction and j-th ground truth.
            continue;
          }
          // Find the maximum overlapped pair.
          float overlap = it->second[j];
          if (overlap >= overlap_threshold && overlap > max_overlap) {
            // If the prediction has not been matched to any ground truth,
            // and the overlap is larger than maximum overlap, update.
            max_gt_idx = j;
            max_overlap = overlap;
          }
        }
        if (max_gt_idx != -1) {
          // Found a matched ground truth.
          CHECK_EQ((*match_indices)[i], -1);
          (*match_indices)[i] = gt_indices[max_gt_idx];
          (*match_overlaps)[i] = max_overlap;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown matching type.";
      break;
  }

  return;
}

void FindMatches(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      vector<map<int, vector<float> > >* all_match_overlaps,
      vector<map<int, vector<int> > >* all_match_indices) {
  // all_match_overlaps->clear();
  // all_match_indices->clear();
  // Get parameters.
 // CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = 21;
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const bool share_location = multibox_loss_param.share_location();
  const int loc_classes = share_location ? 1 : num_classes;
  const MatchType match_type = multibox_loss_param.match_type();
  const float overlap_threshold = multibox_loss_param.overlap_threshold();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  const int background_label_id = multibox_loss_param.background_label_id();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool ignore_cross_boundary_bbox =
      multibox_loss_param.ignore_cross_boundary_bbox();


  // Find the matches.
  int num = all_loc_preds.size();
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > match_indices;
    map<int, vector<float> > match_overlaps;
    // Check if there is ground truth for current image.
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices->push_back(match_indices);
      all_match_overlaps->push_back(match_overlaps);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
    if (!use_prior_for_matching) {
      for (int c = 0; c < loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (!share_location && label == background_label_id) {
          // Ignore background loc predictions.
          continue;
        }
        // Decode the prediction into bbox first.
        vector<NormalizedBBox> loc_bboxes;
        bool clip_bbox = false;
        DecodeBBoxes(prior_bboxes, prior_variances,
                     code_type, encode_variance_in_target, clip_bbox,
                     all_loc_preds[i].find(label)->second, &loc_bboxes);
        MatchBBox(gt_bboxes, loc_bboxes, label, match_type,
                  overlap_threshold, ignore_cross_boundary_bbox,
                  &match_indices[label], &match_overlaps[label]);
      }
    } else {
      // Use prior bboxes to match against all ground truth.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;
      MatchBBox(gt_bboxes, prior_bboxes, label, match_type, overlap_threshold,
                ignore_cross_boundary_bbox, &temp_match_indices,
                &temp_match_overlaps);
      if (share_location) {
    //    LOG(INFO)<<"share_location";

        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
  
      } else {
        // Get ground truth label for each ground truth bbox.
        vector<int> gt_labels;
        for (int g = 0; g < gt_bboxes.size(); ++g) {
          gt_labels.push_back(gt_bboxes[g].label());
        }
        // Distribute the matching results to different loc_class.
        for (int c = 0; c < loc_classes; ++c) {
          if (c == background_label_id) {
            // Ignore background loc predictions.
            continue;
          }
          match_indices[c].resize(temp_match_indices.size(), -1);
          match_overlaps[c] = temp_match_overlaps;
          for (int m = 0; m < temp_match_indices.size(); ++m) {
            if (temp_match_indices[m] > -1) {
              const int gt_idx = temp_match_indices[m];
              CHECK_LT(gt_idx, gt_labels.size());
              if (c == gt_labels[gt_idx]) {
                match_indices[c][m] = gt_idx;
              }
            }
          }
        }
      }
    }
    all_match_indices->push_back(match_indices);
    all_match_overlaps->push_back(match_overlaps);
  }
}



// Explicit initialization.
template void GetPriorBBoxes(const float* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);

int CountNumMatches(const vector<map<int,vector<int> > >& all_match_indices,
  const int num){
  int num_matches = 0;
  for(int i = 0; i < num ; ++i){
    const map<int,vector<int> >& match_indices = all_match_indices[i];
    for(map<int,vector<int> >::const_iterator it = match_indices.begin();
      it != match_indices.end(); ++it){
      const vector<int>& match_index = it->second;
      for (int m = 0;m < match_index.size(); ++m){
        if(match_index[m] > -1){
          ++num_matches;
        }
      }
    }
  }
  return num_matches;
}

inline bool IsEligibleMining(const MiningType mining_type, const int match_idx,
    const float match_overlap, const float neg_overlap) {
  if (mining_type == caffe::MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
    return match_idx == -1 && match_overlap < neg_overlap;
  } else if (mining_type == caffe::MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    return true;
  } else {
    return false;
  }
}

void MineHardExamples(TensorCPU conf_tensor,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices){
  int num = all_loc_preds.size();
  *num_matches = CountNumMatches(*all_match_indices,num);
  *num_negs = 0;
  
  int num_priors = prior_bboxes.size();
  CAFFE_ENFORCE_EQ(num_priors,prior_variances.size());
  const int num_classes = 21;//multibox_loss_param.num_classes()
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool use_prior_for_nms = multibox_loss_param.use_prior_for_nms();
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  const MiningType mining_type = multibox_loss_param.mining_type();

  if(mining_type == caffe::MultiBoxLossParameter_MiningType_NONE){
    return ;
  }
  const LocLossType loc_loss_type = multibox_loss_param.loc_loss_type();
  const float neg_pos_ratio = multibox_loss_param.neg_pos_ratio();
  const float neg_overlap = multibox_loss_param.neg_overlap();
  const bool encode_variance_in_target = multibox_loss_param.encode_variance_in_target();
  const bool has_nms_param = multibox_loss_param.has_nms_param();

  float nms_threshold = 0;
  int top_k = -1;
  if(has_nms_param){
    nms_threshold = multibox_loss_param.nms_param().nms_threshold();
    top_k = multibox_loss_param.nms_param().top_k();
  }

  const int sample_size = multibox_loss_param.sample_size();
  vector<vector<float> > all_conf_loss;
  const float* conf_data = conf_tensor.template data<float>();
  ComputeConfLoss(conf_data ,num ,num_priors ,num_classes ,
    background_label_id,conf_loss_type,*all_match_indices,all_gt_bboxes,
    &all_conf_loss);

  vector<vector<float> >all_loc_loss;

  if(mining_type == caffe::MultiBoxLossParameter_MiningType_HARD_EXAMPLE){

  }else{
    for(int i = 0; i<num ;++i){
      vector<float> loc_loss(num_priors,0.);
      all_loc_loss.push_back(loc_loss);
    }
  }

  for(int i = 0;i < num; ++i){
    map<int, vector<int> > & match_indices = (*all_match_indices)[i];
    const map<int,vector<float> >& match_overlaps = all_match_overlaps[i];
    const vector<float>& conf_loss = all_conf_loss[i];
    const vector<float>& loc_loss = all_loc_loss[i];
    vector<float> loss ;
    std::transform(conf_loss.begin(),conf_loss.end(),loc_loss.begin(),
      std::back_inserter(loss),std::plus<float>());
    set<int> sel_indices;
    vector<int> neg_indices;

    for(map<int,vector<int> >::iterator it = match_indices.begin();
      it != match_indices.end();++it){
      const int label = it->first;
      int num_sel = 0;

      vector<pair<float,int> > loss_indices;
      for(int m = 0; m < match_indices[label].size(); ++m){
        if(IsEligibleMining(mining_type,match_indices[label][m],
          match_overlaps.find(label)->second[m],neg_overlap)){
          loss_indices.push_back(std::make_pair(loss[m],m));
          ++num_sel;
        }
      }
      if(mining_type == caffe::MultiBoxLossParameter_MiningType_MAX_NEGATIVE){
        int num_pos = 0;
        for(int m = 0; m < match_indices[label].size(); ++m){
          if(match_indices[label][m] > -1){
            ++num_pos;
          }
        }
        num_sel = std::min(static_cast<int>(num_pos*neg_pos_ratio),num_sel);
      }

      if(has_nms_param && nms_threshold > 0){

      }else{
        std::sort(loss_indices.begin(),loss_indices.end(),
          SortScorePairDescend<int>);
        for(int n = 0; n < num_sel; ++n){
          sel_indices.insert(loss_indices[n].second);
        }
      }

      for(int m = 0; m < match_indices[label].size(); ++m){
        if(match_indices[label][m] > -1){

        }else if(match_indices[label][m] == -1){
          if(sel_indices.find(m) != sel_indices.end()){
            neg_indices.push_back(m);
            *num_negs += 1;
          }
        }
      }

    }
    all_neg_indices->push_back(neg_indices);
  }

}


void EncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
  const map<int,vector<NormalizedBBox> >&all_gt_bboxes,
  const vector<map<int,vector<int> > >&all_match_indices,
  const vector<NormalizedBBox>& prior_bboxes,
  const vector<vector<float> >& prior_variances,
  const MultiBoxLossParameter& multibox_loss_param,
  float* loc_pred_data,float* loc_gt_data){

  int num = all_loc_preds.size();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target = multibox_loss_param.encode_variance_in_target();
  const bool bp_inside = multibox_loss_param.bp_inside();
  const bool use_prior_for_matching = 
    multibox_loss_param.use_prior_for_matching();
  //LOG(INFO)<<"CodeType = "<<code_type<<"PriorBoxParameter_CodeType_CORNER"
  //<<caffe::PriorBoxParameter_CodeType_CORNER;
  int count = 0;

  for(int i = 0; i < num; ++i){

    for(map<int,vector<int> >::const_iterator 
      it = all_match_indices[i].begin();
      it != all_match_indices[i].end(); ++it){
      const int label = it->first;
      const vector<int>& match_index = it->second;
      CAFFE_ENFORCE(all_loc_preds[i].find(label)!=all_loc_preds[i].end());
      const vector<NormalizedBBox>& loc_pred = 
        all_loc_preds[i].find(label)->second;
      
      for(int j = 0; j < match_index.size(); ++j){
        if(match_index[j]<=-1) continue;
        const int gt_idx = match_index[j];

      //  LOG(INFO)<<"gt_idx = "<<gt_idx;

        CAFFE_ENFORCE(all_gt_bboxes.find(i)!=all_gt_bboxes.end());
        CAFFE_ENFORCE_LT(gt_idx,all_gt_bboxes.find(i)->second.size());
        const NormalizedBBox& gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];
        NormalizedBBox gt_encode;
        CAFFE_ENFORCE_LT(j,prior_bboxes.size());
        EncodeBBox(prior_bboxes[j],prior_variances[j],code_type,
          encode_variance_in_target,gt_bbox,&gt_encode);
        loc_gt_data[count * 4] = gt_encode.xmin();
        loc_gt_data[count * 4 + 1] = gt_encode.ymin();
        loc_gt_data[count * 4 + 2] = gt_encode.xmax();
        loc_gt_data[count * 4 + 3] = gt_encode.ymax();
        CAFFE_ENFORCE_LT(j,loc_pred.size());

        if(bp_inside){
          NormalizedBBox match_bbox = prior_bboxes[j];
          if (!use_prior_for_matching) {
            const bool clip_bbox = false;
            DecodeBBox(prior_bboxes[j], prior_variances[j], code_type,
                       encode_variance_in_target, clip_bbox, loc_pred[j],
                       &match_bbox);
          }
          // When a dimension of match_bbox is outside of image region, use
          // gt_encode to simulate zero gradient.
          loc_pred_data[count * 4] =
              (match_bbox.xmin() < 0 || match_bbox.xmin() > 1) ?
              gt_encode.xmin() : loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] =
              (match_bbox.ymin() < 0 || match_bbox.ymin() > 1) ?
              gt_encode.ymin() : loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] =
              (match_bbox.xmax() < 0 || match_bbox.xmax() > 1) ?
              gt_encode.xmax() : loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] =
              (match_bbox.ymax() < 0 || match_bbox.ymax() > 1) ?
              gt_encode.ymax() : loc_pred[j].ymax();
        }else{
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();           
        }

        if (encode_variance_in_target) {
          for (int k = 0; k < 4; ++k) {
            CHECK_GT(prior_variances[j][k], 0);
            loc_pred_data[count * 4 + k] /= prior_variances[j][k];
            loc_gt_data[count * 4 + k] /= prior_variances[j][k];
          }
        }

        ++count;
      }
      
    }

  }

}


void EncodeConfPrediction(const float* conf_data,const int num,
  const int num_priors,const MultiBoxLossParameter& multibox_loss_param,
  const vector<map<int,vector<int> > >&all_match_indices,
  const vector<vector<int> >& all_neg_indices,
  const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
  float* conf_pred_data,float* conf_gt_data){

  const int num_classes = 21;
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool map_object_to_agnostic = 
    multibox_loss_param.map_object_to_agnostic();
  if(map_object_to_agnostic){
    if(background_label_id>=0){
      CAFFE_ENFORCE_EQ(num_classes, 2);
    }else{
      CAFFE_ENFORCE_EQ(num_classes, 1);
    }
  }
  const MiningType mining_type = multibox_loss_param.mining_type();
  bool do_neg_mining;
  if(multibox_loss_param.has_do_neg_mining()){
    LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
    do_neg_mining = multibox_loss_param.do_neg_mining();
    CAFFE_ENFORCE_EQ(do_neg_mining,mining_type!=caffe::MultiBoxLossParameter_MiningType_NONE);
  }

  do_neg_mining = mining_type != caffe::MultiBoxLossParameter_MiningType_NONE;
//  LOG(INFO)<<"do_neg_mining = "<<do_neg_mining;
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  int count = 0;
  for(int i = 0; i < num; ++i){
    if(all_gt_bboxes.find(i) != all_gt_bboxes.end()){
      const map<int,vector<int> >&match_indices = all_match_indices[i];

      for(map<int,vector<int> >::const_iterator it = 
        match_indices.begin();it!=match_indices.end();++it){
        const vector<int>& match_index = it->second;
        CAFFE_ENFORCE_EQ(match_index.size(),num_priors);
        for(int j = 0; j < num_priors; ++j){
          if(match_index[j] <= -1){
            continue;
          }
          const int gt_label = map_object_to_agnostic ?
            background_label_id + 1 : all_gt_bboxes.find(i)->second[match_index[j]].label();
          int idx = do_neg_mining ? count : j ;
          if(conf_loss_type == caffe::MultiBoxLossParameter_ConfLossType_SOFTMAX){
            conf_gt_data[idx] = gt_label;
          }else{
            LOG(FATAL) << "Unknown conf loss type.";
          }
          if(do_neg_mining){
             memcpy(conf_pred_data + count * num_classes , conf_data + j * num_classes,
               num_classes*sizeof(float));
            // for(int k = 0;k<num_classes;++k)
            //   conf_pred_data[count*num_classes + k]=conf_data[j*num_classes +k];
            ++count;
          }
        }
      }
    }

    if(do_neg_mining){
      for (int n = 0; n < all_neg_indices[i].size(); ++n){
        int j = all_neg_indices[i][n];
        CAFFE_ENFORCE_LT(j,num_priors);
         memcpy(conf_pred_data + count * num_classes , conf_data + j * num_classes,
               num_classes*sizeof(float));
        
        if(conf_loss_type == caffe:: MultiBoxLossParameter_ConfLossType_SOFTMAX){
          conf_gt_data[count] = background_label_id;
        }else{
          LOG(FATAL)<<"Unknown conf loss type.";
        }
        ++count;
      }
    }

    if(do_neg_mining){
      conf_data += num_priors * num_classes;
    }else{
      conf_data += num_priors;
    }

  }

}

void ComputeConfLoss(const float* conf_data,const int num,
  const int num_preds_per_class,const int num_classes,
  const int background_label_id,const ConfLossType loss_type,
  const vector<map<int,vector<int> > >& all_match_indices,
  const map<int ,vector<NormalizedBBox> >& all_gt_bboxes,
  vector<vector<float> >* all_conf_loss){
  CAFFE_ENFORCE_LT(background_label_id,num_classes);
  all_conf_loss->clear();
  for(int i = 0; i < num; i++){
    vector<float> conf_loss;
    const map<int,vector<int> >& match_indices = all_match_indices[i];
    for(int p = 0;p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      int label = background_label_id ;

      for(map<int,vector<int> >::const_iterator it = 
        match_indices.begin();it != match_indices.end();++it){
        const vector<int>& match_index = it->second;
        CAFFE_ENFORCE_EQ(match_index.size(),num_preds_per_class);
        if(match_index[p]>-1){
          CAFFE_ENFORCE(all_gt_bboxes.find(i) != all_gt_bboxes.end());
          const vector<NormalizedBBox>& gt_bboxes = 
          all_gt_bboxes.find(i)->second;
          CAFFE_ENFORCE_LT(match_index[p],gt_bboxes.size());
          label = gt_bboxes[match_index[p]].label();
          CAFFE_ENFORCE_GE(label,0);
          CAFFE_ENFORCE_NE(label,background_label_id);
          CAFFE_ENFORCE_LT(label,num_classes);
          break;
        }
      }

      float loss = 0;
      if(loss_type == caffe::MultiBoxLossParameter_ConfLossType_SOFTMAX){
        CAFFE_ENFORCE_GE(label,0);
        CAFFE_ENFORCE_LT(label,num_classes);
        float maxval = -FLT_MAX;
        for(int c = 0; c < num_classes; ++c){
          maxval = std::max(conf_data[start_idx + c],maxval);
        }
        float sum = 0.;
        for(int c = 0; c < num_classes; ++c){
          sum+= std::exp(conf_data[start_idx + c] - maxval);
        }
        float prob = std::exp(conf_data[start_idx + label] - maxval )/sum;
        loss = -log(std::max(prob,float(FLT_MIN)));
      }else{
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}

/*
template void MineHardExamples(TensorCPU conf_tensor,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);
*/


float BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
    }
  }
}

void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, vector<LabelBBox>* all_decode_bboxes) {
  CAFFE_ENFORCE_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
    // Decode predictions into bboxes.
    LabelBBox& decode_bboxes = (*all_decode_bboxes)[i]; 
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c; 
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      const vector<NormalizedBBox>& label_loc_preds =
          all_loc_preds[i].find(label)->second;
      DecodeBBoxes(prior_bboxes, prior_variances,
                   code_type, variance_encoded_in_target, clip,
                   label_loc_preds, &(decode_bboxes[label]));
    }
  }
}

void DecodeBBoxes(
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
    vector<NormalizedBBox>* decode_bboxes) {
  CAFFE_ENFORCE_EQ(prior_bboxes.size(), prior_variances.size()); // size = 8732
  CAFFE_ENFORCE_EQ(prior_bboxes.size(), bboxes.size()); // 
  int num_bboxes = prior_bboxes.size();
  if (num_bboxes >= 1) {
  	CAFFE_ENFORCE_EQ(prior_variances[0].size(), 4);
  }
  decode_bboxes->clear();
  for (int i = 0; i < num_bboxes; ++i) { // 8732
    NormalizedBBox decode_bbox;
    DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
               variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
    decode_bboxes->push_back(decode_bbox);
  }
}
void DecodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const NormalizedBBox& bbox,
    NormalizedBBox* decode_bbox) {
  if (code_type == caffe::PriorBoxParameter_CodeType_CORNER) {
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(
          prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
      decode_bbox->set_ymin(
          prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
      decode_bbox->set_xmax(
          prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
      decode_bbox->set_ymax(
          prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
    }
  } else if (code_type == caffe::PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CAFFE_ENFORCE_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CAFFE_ENFORCE_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width = exp(bbox.xmax()) * prior_width;
      decode_bbox_height = exp(bbox.ymax()) * prior_height;
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y =
          prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width =
          exp(prior_variance[2] * bbox.xmax()) * prior_width;
      decode_bbox_height =
          exp(prior_variance[3] * bbox.ymax()) * prior_height;
    }

    decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
    decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
    decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
    decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
  } else if (code_type == caffe::PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CAFFE_ENFORCE_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CAFFE_ENFORCE_GT(prior_height, 0);
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin() * prior_width);
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin() * prior_height);
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax() * prior_width);
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax() * prior_height);
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(
          prior_bbox.xmin() + prior_variance[0] * bbox.xmin() * prior_width);
      decode_bbox->set_ymin(
          prior_bbox.ymin() + prior_variance[1] * bbox.ymin() * prior_height);
      decode_bbox->set_xmax(
          prior_bbox.xmax() + prior_variance[2] * bbox.xmax() * prior_width);
      decode_bbox->set_ymax(
          prior_bbox.ymax() + prior_variance[3] * bbox.ymax() * prior_height);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
  float bbox_size = BBoxSize(*decode_bbox);
  decode_bbox->set_size(bbox_size);
  if (clip_bbox) {
    ClipBBox(*decode_bbox, decode_bbox);
  }
}


void EncodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const CodeType code_type, const bool encode_variance_in_target,
    const NormalizedBBox& bbox, NormalizedBBox* encode_bbox) {
  if (code_type == caffe::PriorBoxParameter_CodeType_CORNER) {
    if (encode_variance_in_target) {
      encode_bbox->set_xmin(bbox.xmin() - prior_bbox.xmin());
      encode_bbox->set_ymin(bbox.ymin() - prior_bbox.ymin());
      encode_bbox->set_xmax(bbox.xmax() - prior_bbox.xmax());
      encode_bbox->set_ymax(bbox.ymax() - prior_bbox.ymax());
    } else {
      // Encode variance in bbox.
      CHECK_EQ(prior_variance.size(), 4);
      for (int i = 0; i < prior_variance.size(); ++i) {
        CHECK_GT(prior_variance[i], 0);
      }
      encode_bbox->set_xmin(
          (bbox.xmin() - prior_bbox.xmin()) / prior_variance[0]);
      encode_bbox->set_ymin(
          (bbox.ymin() - prior_bbox.ymin()) / prior_variance[1]);
      encode_bbox->set_xmax(
          (bbox.xmax() - prior_bbox.xmax()) / prior_variance[2]);
      encode_bbox->set_ymax(
          (bbox.ymax() - prior_bbox.ymax()) / prior_variance[3]);
    }
  } else if (code_type == caffe::PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CAFFE_ENFORCE_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CAFFE_ENFORCE_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;
 
    float bbox_width = bbox.xmax() - bbox.xmin();
    CAFFE_ENFORCE_GT(bbox_width, 0);
    float bbox_height = bbox.ymax() - bbox.ymin();
    CAFFE_ENFORCE_GT(bbox_height, 0);
    float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.;
    float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.;

    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width);
      encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height);
      encode_bbox->set_xmax(log(bbox_width / prior_width));
      encode_bbox->set_ymax(log(bbox_height / prior_height));
    } else {
      // Encode variance in bbox.
      encode_bbox->set_xmin(
          (bbox_center_x - prior_center_x) / prior_width / prior_variance[0]);
      encode_bbox->set_ymin(
          (bbox_center_y - prior_center_y) / prior_height / prior_variance[1]);
      encode_bbox->set_xmax(
          log(bbox_width / prior_width) / prior_variance[2]);
      encode_bbox->set_ymax(
          log(bbox_height / prior_height) / prior_variance[3]);
    }
  } else if (code_type == caffe::PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) / prior_width);
      encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) / prior_height);
      encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) / prior_width);
      encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) / prior_height);
    } else {
      // Encode variance in bbox.
      CHECK_EQ(prior_variance.size(), 4);
      for (int i = 0; i < prior_variance.size(); ++i) {
        CHECK_GT(prior_variance[i], 0);
      }
      encode_bbox->set_xmin(
          (bbox.xmin() - prior_bbox.xmin()) / prior_width / prior_variance[0]);
      encode_bbox->set_ymin(
          (bbox.ymin() - prior_bbox.ymin()) / prior_height / prior_variance[1]);
      encode_bbox->set_xmax(
          (bbox.xmax() - prior_bbox.xmax()) / prior_width / prior_variance[2]);
      encode_bbox->set_ymax(
          (bbox.ymax() - prior_bbox.ymax()) / prior_height / prior_variance[3]);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
}

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}

void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
              NormalizedBBox* clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), width), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), height), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), width), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), height), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}


void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices) {
  // Sanity check.
  CAFFE_ENFORCE_EQ(bboxes.size(), scores.size());

  // Get top_k scores (with corresponding indices).
  vector<pair<float, int> > score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}
void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}
template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);


float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                     const bool normalized) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_width, intersect_height;
  if (normalized) {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
  } else {
    intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
    intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
  }
  if (intersect_width > 0 && intersect_height > 0) {
    float intersect_size = intersect_width * intersect_height;
    float bbox1_size = BBoxSize(bbox1);
    float bbox2_size = BBoxSize(bbox2);
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
  } else {
    return 0.;
  }
}
void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}

template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices) {
  // Get top_k scores (with corresponding indices).
  vector<pair<Dtype, int> > score_index_vec;
  GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template
void ApplyNMSFast(const float* bboxes, const float* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);
template
void ApplyNMSFast(const double* bboxes, const double* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);


template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < num; ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::sort(score_index_vec->begin(), score_index_vec->end(),
            SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template
void GetMaxScoreIndex(const float* scores, const int num, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);
template
void GetMaxScoreIndex(const double* scores, const int num,
      const float threshold, const int top_k,
      vector<pair<double, int> >* score_index_vec);

template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSize(bbox1);
    const Dtype bbox2_size = BBoxSize(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template float JaccardOverlap(const float* bbox1, const float* bbox2);
template double JaccardOverlap(const double* bbox1, const double* bbox2);

template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template float BBoxSize(const float* bbox, const bool normalized);
template double BBoxSize(const double* bbox, const bool normalized);

float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_size = BBoxSize(intersect_bbox);
  if (intersect_size > 0) {
    float bbox1_size = BBoxSize(bbox1);
    return intersect_size / bbox1_size;
  } else {
    return 0.;
  }
}

void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                NormalizedBBox* loc_bbox) {
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  loc_bbox->set_xmin(src_bbox.xmin() + bbox.xmin() * src_width);
  loc_bbox->set_ymin(src_bbox.ymin() + bbox.ymin() * src_height);
  loc_bbox->set_xmax(src_bbox.xmin() + bbox.xmax() * src_width);
  loc_bbox->set_ymax(src_bbox.ymin() + bbox.ymax() * src_height);
  loc_bbox->set_difficult(bbox.difficult());
}

bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
                        const NormalizedBBox& bbox,
                        const EmitConstraintParam& emit_constraint) {
  string emit_type = emit_constraint.emit_type;
  if (emit_type == "CENTER") {
    float x_center = (bbox.xmin() + bbox.xmax()) / 2;
    float y_center = (bbox.ymin() + bbox.ymax()) / 2;
    if (x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax() &&
        y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax()) {
      return true;
    } else {
      return false;
    }
  } else if (emit_type == "MIN_OVERLAP") {
//    float bbox_coverage = BBoxCoverage(bbox, src_bbox);
//    return bbox_coverage > emit_constraint.emit_overlap;
    return false;
  } else {
    LOG(FATAL) << "Unknown emit type.";
    return false;
  }
}

bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox) {
  if (bbox.xmin() >= src_bbox.xmax() || bbox.xmax() <= src_bbox.xmin() ||
      bbox.ymin() >= src_bbox.ymax() || bbox.ymax() <= src_bbox.ymin()) {
    return false;
  }
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
  proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
  proj_bbox->set_difficult(bbox.difficult());
  ClipBBox(*proj_bbox, proj_bbox);
  if (BBoxSize(*proj_bbox) > 0) {
    return true;
  } else {
    return false;
  }
}

void ExtrapolateBBox(const ResizeParam& param, const int height,
    const int width, const NormalizedBBox& crop_bbox, NormalizedBBox* bbox) {
  float height_scale = param.height_scale;
  float width_scale = param.width_scale;
  /*
  if (height_scale > 0 && width_scale > 0 &&
      param.resize_mode() == ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
    float orig_aspect = static_cast<float>(width) / height;
    float resize_height = param.height();
    float resize_width = param.width();
    float resize_aspect = resize_width / resize_height;
    if (orig_aspect < resize_aspect) {
      resize_height = resize_width / orig_aspect;
    } else {
      resize_width = resize_height * orig_aspect;
    }
    float crop_height = resize_height * (crop_bbox.ymax() - crop_bbox.ymin());
    float crop_width = resize_width * (crop_bbox.xmax() - crop_bbox.xmin());
    CHECK_GE(crop_width, width_scale);
    CHECK_GE(crop_height, height_scale);
    bbox->set_xmin(bbox->xmin() * crop_width / width_scale);
    bbox->set_xmax(bbox->xmax() * crop_width / width_scale);
    bbox->set_ymin(bbox->ymin() * crop_height / height_scale);
    bbox->set_ymax(bbox->ymax() * crop_height / height_scale);
  }
  */
}

void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox) {
  scale_bbox->set_xmin(bbox.xmin() * width);
  scale_bbox->set_ymin(bbox.ymin() * height);
  scale_bbox->set_xmax(bbox.xmax() * width);
  scale_bbox->set_ymax(bbox.ymax() * height);
  scale_bbox->clear_size();
  bool normalized = !(width > 1 || height > 1);
  scale_bbox->set_size(BBoxSize(*scale_bbox, normalized));
  scale_bbox->set_difficult(bbox.difficult());
}

void OutputBBox(const NormalizedBBox& bbox, const pair<int, int>& img_size,
                const bool has_resize, const ResizeParam& resize_param,
                NormalizedBBox* out_bbox) {
  const int height = img_size.first;
  const int width = img_size.second;
  NormalizedBBox temp_bbox = bbox;
  if (has_resize && resize_param.resize_valid) {
    float resize_height = resize_param.height;
    CHECK_GT(resize_height, 0);
    float resize_width = resize_param.width;
    CHECK_GT(resize_width, 0);
    float resize_aspect = resize_width / resize_height;
    int height_scale = resize_param.height_scale;
    int width_scale = resize_param.width_scale;
    float aspect = static_cast<float>(width) / height;

    float padding;
    NormalizedBBox source_bbox;
    
    if(resize_param.resize_mode == "WARP"){
      ClipBBox(temp_bbox, &temp_bbox);
      ScaleBBox(temp_bbox, height, width, out_bbox);
    }
    else if(resize_param.resize_mode == "FIT_LARGE_SIZE_AND_PAD"){
      if (aspect > resize_aspect) {
        padding = (resize_height - resize_width / aspect) / 2;
        source_bbox.set_xmin(0.);
        source_bbox.set_ymin(padding / resize_height);
        source_bbox.set_xmax(1.);
        source_bbox.set_ymax(1. - padding / resize_height);
      } else {
        padding = (resize_width - resize_height * aspect) / 2;
        source_bbox.set_xmin(padding / resize_width);
        source_bbox.set_ymin(0.);
        source_bbox.set_xmax(1. - padding / resize_width);
        source_bbox.set_ymax(1.);
      }
      ProjectBBox(source_bbox, bbox, &temp_bbox);
      ClipBBox(temp_bbox, &temp_bbox);
      ScaleBBox(temp_bbox, height, width, out_bbox);
    }
    else if(resize_param.resize_mode == "FIT_SMALL_SIZE"){
      if (height_scale == 0 || width_scale == 0) {
        ClipBBox(temp_bbox, &temp_bbox);
        ScaleBBox(temp_bbox, height, width, out_bbox);
      } else {
        ScaleBBox(temp_bbox, height_scale, width_scale, out_bbox);
        ClipBBox(*out_bbox, height, width, out_bbox);
      }
    }
    else{
      LOG(FATAL) << "Unknown resize mode.";
    }
    
  } else {
    // Clip the normalized bbox first.
    ClipBBox(temp_bbox, &temp_bbox);
    // Scale the bbox according to the original image size.
    ScaleBBox(temp_bbox, height, width, out_bbox);
  }
}

template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections) {
  all_detections->clear();
 // LOG(INFO)<<"num_det = "<<num_det;
  for (int i = 0; i < num_det; ++i) {
    int start_idx = i * 7;
    int item_id = det_data[start_idx];
 //   LOG(INFO)<<"item_id = "<<item_id;
    if (item_id == -1) {
      continue;
    }
    int label = det_data[start_idx + 1];
    
    CHECK_NE(background_label_id, label)
        << "Found background label in the detection results.";
    NormalizedBBox bbox;
    bbox.set_score(det_data[start_idx + 2]);
    bbox.set_xmin(det_data[start_idx + 3]);
    bbox.set_ymin(det_data[start_idx + 4]);
    bbox.set_xmax(det_data[start_idx + 5]);
    bbox.set_ymax(det_data[start_idx + 6]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_detections)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void GetDetectionResults(const float* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections);
template void GetDetectionResults(const double* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections);

} // namespace caffe2
