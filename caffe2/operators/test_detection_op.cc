#include "caffe2/operators/test_detection_op.h"




namespace caffe2{

bool SortScorePairDescend(const pair<float, int>& pair1,
                          const pair<float, int>& pair2) {
  return pair1.first > pair2.first;
}

void CumSum(const vector<pair<float, int> >&pairs , vector<int>* cumsum) {
	vector<pair<float, int> > sort_pairs = pairs;
	std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
					SortScorePairDescend);

	cumsum->clear();
	for (int i = 0; i < sort_pairs.size(); ++i){
		if(i == 0){
			cumsum->push_back(sort_pairs[i].second);
		}else{
			cumsum->push_back(cumsum->back() + sort_pairs[i].second);
		}
	}
}

void ComputeAP(const vector<pair<float, int> >&tp, const int num_pos,
				const vector<pair<float, int> >&fp, const string ap_version,
				vector<float>* prec, vector<float>* rec, float* ap){
	const float eps = 1e-6;
	CAFFE_ENFORCE_EQ(tp.size(), fp.size(), "tp must have same size as fp.");
	const int num = tp.size();
	for (int i = 0; i < num; ++i){
		CAFFE_ENFORCE_LE(fabs(tp[i].first - fp[i].first), eps);
		CAFFE_ENFORCE_EQ(tp[i].second, 1 - fp[i].second);
	}
	prec->clear();
	rec->clear();
	*ap = 0;
	if (tp.size() == 0 || num_pos == 0){
		return;
	}

	vector<int> tp_cumsum;
	CumSum(tp, &tp_cumsum);
	CAFFE_ENFORCE_EQ(tp_cumsum.size(), num);

	vector<int> fp_cumsum;
	CumSum(fp, &fp_cumsum);
	CAFFE_ENFORCE_EQ(fp_cumsum.size(), num);

	for (int i = 0; i < num; ++i){
		prec->push_back(static_cast<float>(tp_cumsum[i]) / 
			(tp_cumsum[i] + fp_cumsum[i]));
	}

	for (int i = 0; i < num; ++i){
		CAFFE_ENFORCE_LE(tp_cumsum[i], num_pos);
		rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
	}

	if (ap_version == "11point"){
		vector<float> max_precs(11, 0.);
		int start_idx = num - 1;
		for (int j = 10; j >= 0; --j){
			for (int i = start_idx; i >= 0; --i){
				if ((*rec)[i] < j / 10.){
					start_idx = i;
					if (j > 0){
						max_precs[j - 1] = max_precs[j];
					}
					break;
				}else{
					if(max_precs[j] < (*prec)[i]){
						max_precs[j] = (*prec)[i];
					}
				}
			}
		}
		for (int j = 10; j >= 0; --j){
			*ap += max_precs[j] / 11.;
		}
	}else{
		LOG(FATAL) <<"Unknown ap_version: " <<ap_version;
	}
}

template <>
bool TestDetectionOp<float, CPUContext>::RunOnDevice()
{
	auto& result = Input(0);
	auto* mAp = Output(0);

	CAFFE_ENFORCE_EQ(result.dim(3), 5);

	int num_det = result.dim(2);

	const float* result_vec = result.data<float>();

	count_ += iter_size_;
	for (int j = 0; j < result.dim(0); ++j){
		for(int k = 0; k < num_det; ++k){
			int item_id = static_cast<int>(result_vec[k * 5]);
			int label = static_cast<int>(result_vec[k * 5 + 1]);
			if(item_id == -1){
				if(all_num_pos[j].find(label) == all_num_pos[j].end()){
					all_num_pos[j][label] = static_cast<int>(result_vec[k * 5 + 2]);
				}else{
					all_num_pos[j][label] += static_cast<int>(result_vec[k * 5 + 2]);
				}
			}else{
				float score = result_vec[k * 5 +2];
				int tp = static_cast<int>(result_vec[k * 5 + 3]);
				int fp = static_cast<int>(result_vec[k * 5 + 4]);
				if( tp == 0 && fp == 0){
					continue;
				}
				all_true_pos[j][label].push_back(std::make_pair(score, tp));
				all_false_pos[j][label].push_back(std::make_pair(score, fp));
			}
		}
	}
	if(count_ >= test_size_){
		for(int i = 0; i < all_true_pos.size(); ++i){
			if (all_true_pos.find(i) == all_true_pos.end()){
				LOG(INFO) << "Missing output_blob true_pos: " << i;
			}
			const map<int, vector<pair<float, int> > >& true_pos =
				all_true_pos.find(i)->second;
			if(all_false_pos.find(i) == all_false_pos.end()){
				LOG(INFO) << "Missing output_blob flase_pos: " << i;
			}
			const map<int, vector<pair<float, int> > >& false_pos =
				all_false_pos.find(i)->second;
			if (all_num_pos.find(i) == all_num_pos.end()){
				LOG(INFO) << "Missing output_blob num_pos: " << i;
			}
			const map<int, int>& num_pos = all_num_pos.find(i)->second;
			map<int, float> APs;
			float mAP = 0.;

			for (map<int, int>::const_iterator it = num_pos.begin();
				it != num_pos.end(); ++it){
				int label = it->first;
				int label_num_pos = it->second;
				if (true_pos.find(label) == true_pos.end()){
					LOG(INFO) << "Missing true_pos for label: " << label;
					continue;
				}
				const vector<pair<float, int> >& label_true_pos = 
					true_pos.find(label)->second;
				if(false_pos.find(label) == false_pos.end()){
					LOG(INFO) << "Missing false_pos for label :" << label;
				}

				const vector<pair<float, int> >& label_false_pos = 
					false_pos.find(label)->second;
				vector<float> prec, rec;
				ComputeAP(label_false_pos, label_num_pos, label_false_pos,
						ap_version_, &prec, &rec, &(APs[label]));
				mAP += APs[label];

			}
			mAP /= num_pos.size();
		}
	}


	return true;
}

namespace{

REGISTER_CPU_OPERATOR(TestDetection,TestDetectionOp<float,CPUContext>);
NO_GRADIENT(TestDetection);

OPERATOR_SCHEMA(TestDetection)
	.NumInputs(1)
	.NumOutputs(1);

}

}