#include "caffe2/operators/detection_evalute_op.h"

namespace caffe2{
template <>
bool DetectionEvaluteOp<float, CPUContext>::RunOnDevice()
{
    auto& det = Input(0);
    auto& gt = Input(1);
    auto* det_eval = Output(0); 
    const float* det_data = det.data<float>();
    const float* gt_data = gt.data<float>();
    
    CAFFE_ENFORCE_LE(count_, sizes_.size());
    CAFFE_ENFORCE_EQ(det.dim(0), 1);
    CAFFE_ENFORCE_EQ(det.dim(1), 1);
    CAFFE_ENFORCE_EQ(det.dim(3), 7);
    CAFFE_ENFORCE_EQ(gt.dim(0), 1);
    CAFFE_ENFORCE_EQ(gt.dim(1), 1);
    CAFFE_ENFORCE_EQ(gt.dim(3), 8);
	
    int num_pos_classes = background_label_id_ == -1 ?
      num_classes_ : num_classes_ - 1;
    int num_valid_det = 0;
    for(int i = 0; i < det.dim(2); ++i){
	if(det_data[i * 7 + 1] != -1){
	    ++num_valid_det;
        }
    }
//    LOG(INFO)<<"num_valid_det = "<<num_valid_det <<"num_pos_classes = "<<num_pos_classes;
    det_eval->Resize(TIndex(1),TIndex(1),TIndex(num_pos_classes + num_valid_det),TIndex(5));

    float* det_eval_data = det_eval->mutable_data<float>();
    
   
    map<int, LabelBBox> all_detections;
    GetDetectionResults(det_data, det.dim(2), background_label_id_,
			&all_detections);  // item_id <label , bbox>  

    map<int, LabelBBox> all_gt_bboxes;
    GetGroundTruth(gt_data, gt.dim(2), background_label_id_,
      true, &all_gt_bboxes);// item_id <label, bbox>
    
    math::Set<float,CPUContext>(det_eval->size(), 0., det_eval_data, &context_);
    int num_det=0;

    map<int, int> num_pos;// bbox label  -> num 
    for (map<int, LabelBBox>::iterator it = all_gt_bboxes.begin();
        it != all_gt_bboxes.end(); ++it){
	for(LabelBBox::iterator iit = it->second.begin(); iit != it->second.end();
		++iit){
        int count = 0;
            if (evaluate_difficult_gt_){
    	       count = iit->second.size();
            }else{
        	   for (int i=0; i < iit->second.size(); ++i){
        		if(!iit->second[i].difficult()){
        		  ++count;
        		}
        	   } 
            }
        	if (num_pos.find(iit->first) == num_pos.end()){
        	  num_pos[iit->first] = count;
        	}else{
        	  num_pos[iit->first] += count;
        	}

        }
    }   
 
    for (int c = 0; c < num_classes_; ++c){
    	if (c == background_label_id_){continue;}
    	det_eval_data[num_det * 5] = -1;
            det_eval_data[num_det * 5 + 1] = c;
    	if (num_pos.find(c) == num_pos.end()){
    	   det_eval_data[num_det * 5 + 2] = 0;
    	}else{
    	   det_eval_data[num_det * 5 + 2] = num_pos.find(c)->second;
    	}
            det_eval_data[num_det * 5 + 3] = -1;
            det_eval_data[num_det * 5 + 4] = -1;
        ++num_det;
    }

    for (map<int, LabelBBox>::iterator it = all_detections.begin();
        it != all_detections.end(); ++it){
        int image_id = it->first;
      //  LOG(INFO)<<"image_id = "<<image_id;
        LabelBBox& detecions = it->second;
        if (all_gt_bboxes.find(image_id) == all_gt_bboxes.end()){
            for (LabelBBox:: iterator iit = detecions.begin();
                iit != detecions.end(); ++iit){
                int label = iit->first;
                if (label == -1){continue;}
                const vector<NormalizedBBox>& bboxes = iit->second;
                for (int i = 0; i < bboxes.size(); ++i){
                    det_eval_data[num_det * 5] = image_id;
                    det_eval_data[num_det * 5 + 1] = label;
                    det_eval_data[num_det * 5 + 2] = bboxes[i].score();
                    det_eval_data[num_det * 5 + 3] = 0;
                    det_eval_data[num_det * 5 + 4] = 1;
                    ++num_det;
                }
            }
        }else{
       //     LOG(INFO)<<"use_normalized_box = "<<use_normalized_bbox_;
            LabelBBox& label_bboxes = all_gt_bboxes.find(image_id)->second;
            for (LabelBBox::iterator iit = detecions.begin();
                iit != detecions.end(); ++iit){
                int label = iit->first;
		//LOG(INFO)<<"label = "<<label;
                if (label == -1){continue;}
                vector<NormalizedBBox>& bboxes = iit->second;
                if (label_bboxes.find(label) == label_bboxes.end()){
                    for (int i = 0; i < bboxes.size(); ++i){
                        det_eval_data[num_det * 5] = image_id;
                        det_eval_data[num_det * 5 + 1] = label;
                        det_eval_data[num_det * 5 + 2] = bboxes[i].score();
                        det_eval_data[num_det * 5 + 3] = 0;
                        det_eval_data[num_det * 5 + 4] = 1;
                        ++num_det;
                    }
                }else{
                    vector<NormalizedBBox>& gt_bboxes = label_bboxes.find(label)->second;
                    if(!use_normalized_bbox_){
                        CAFFE_ENFORCE_LT(count_,sizes_.size());
                        for (int i = 0; i < gt_bboxes.size(); ++i){
                            OutputBBox(gt_bboxes[i], sizes_[count_], has_resize_,
                                resize_param_, &(gt_bboxes[i]));
                        }
                    }
                    vector<bool> visited(gt_bboxes.size(), false);
                    std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);
                    for (int i = 0; i < bboxes.size(); ++i){
                        det_eval_data[num_det * 5] = image_id;
                        det_eval_data[num_det * 5 + 1] = label;
                        det_eval_data[num_det * 5 + 2] = bboxes[i].score();
                        if(!use_normalized_bbox_){         
                            OutputBBox(bboxes[i], sizes_[count_], has_resize_,
                                resize_param_, &(bboxes[i]));
                        }
                        float overlap_max = -1.;
                        int jmax = -1;
                        for (int j = 0; j < gt_bboxes.size(); ++j){
                            float overlap = JaccardOverlap(bboxes[i], gt_bboxes[j], use_normalized_bbox_);
                            if(overlap > overlap_max){
                                overlap_max = overlap;
                                jmax = j;
                            }
                        }  
		//	LOG(INFO)<<"overlap_max = "<<overlap_max;
                        if (overlap_max >= overlap_threshold_){
                            if (evaluate_difficult_gt_ || (!evaluate_difficult_gt_ && !gt_bboxes[jmax].difficult())){
                                if (!visited[jmax]){
                                    det_eval_data[num_det * 5 + 3] = 1;
                                    det_eval_data[num_det * 5 + 4] = 0;
                                    visited[jmax] = true;
                                }else{
                                    det_eval_data[num_det * 5 + 3] = 0;
                                    det_eval_data[num_det * 5 + 4] = 1;
                                }
                            }
                        }else{
                            det_eval_data[num_det * 5 + 3] = 0;
                            det_eval_data[num_det * 5 + 4] = 1;
                        }
                        ++num_det;
                    }
                }
            }
        }
        if (sizes_.size() > 0){
            ++count_;
            if(count_ == sizes_.size()){
                count_;
            }
        }
    }
    return true;
}

namespace{
REGISTER_CPU_OPERATOR(DetectionEvalute,DetectionEvaluteOp<float,CPUContext>);
NO_GRADIENT(DetectionEvalute);

OPERATOR_SCHEMA(DetectionEvalute)
	.NumInputs(2)
	.NumOutputs(1)
	.SetDoc(R"DOC()DOC");

}

}
