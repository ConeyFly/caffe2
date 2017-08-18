#ifndef CAFFE2_MULTIBOX_LOSS_OP_H_
#define CAFFE2_MULTIBOX_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"
#include "caffe2/utils/bbox_util.h"
#include "caffe2/operators/softmax_shared.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/context_gpu.h"
using namespace std;

namespace caffe2{

using caffe::NormalizedBBox;
using caffe::MultiBoxLossParameter;

template<class Context>
class MultiboxLossBaseOp :public Operator<Context>{
public:
	USE_OPERATOR_CONTEXT_FUNCTIONS;
	MultiboxLossBaseOp(const OperatorDef& operator_def, Workspace* ws)
		:Operator<Context>(operator_def, ws){

		}

	bool RunOnDevice() override{
		return RunOnDeviceImpl();
	}

	virtual bool RunOnDeviceImpl(){
		CAFFE_NOT_IMPLEMENTED;
	}

protected:
	
};


template <typename T , class Context>
class MultiboxLossOp final : public Operator<Context>{
public:
	MultiboxLossOp(const OperatorDef& operator_def,Workspace* ws)
		:Operator<Context>(operator_def,ws),
		 num_classes_(OperatorBase::GetSingleArgument<int>("num_classes",21)),
		 share_location_(OperatorBase::GetSingleArgument<bool>("share_location",true)),
		 scale_(OperatorBase::GetSingleArgument<float>("scale",1.0)),
		 background_label_id_(OperatorBase::GetSingleArgument<int>("background_label_id",0))
		{
			loc_classes_ = share_location_ ? 1 : num_classes_;
			loc_loss_type_ = multibox_loss_param_.loc_loss_type();
			conf_loss_type_ = multibox_loss_param_.conf_loss_type();
			do_neg_mining_ = true;
		}

	USE_OPERATOR_CONTEXT_FUNCTIONS;

	bool RunOnDevice() override;
protected:
	float loc_weight_;
	int num_priors_;
	int loc_classes_;
	int num_gt_;
	int num_;
	int num_classes_;
	int background_label_id_;
	bool share_location_;
	bool use_difficult_gt_;
	bool do_neg_mining_;
	int num_matches_ ;
	int num_conf_;
//	vector<map<int, vector<int> > > all_match_indices_;
//	vector<vector<int> > all_neg_indices_;

	MultiBoxLossParameter multibox_loss_param_;

	ConfLossType conf_loss_type_;
	LocLossType loc_loss_type_;

	Tensor<Context> loc_pred_;
	Tensor<Context> loc_gt_;
	Tensor<Context> loc_loss_;
	Tensor<Context> diff_;
	Tensor<Context> errors_;

	float scale_;
	Tensor<Context> conf_pred_;
	Tensor<Context> conf_gt_;
	Tensor<Context> conf_loss_;
	Tensor<Context> rowmax_;
	Tensor<Context> loss_;
	Tensor<Context> sum_multiplier_;
	Tensor<Context> prob;
	vector<map<int,vector<int> > > all_match_indices_;
	vector<vector<int> > all_neg_indices_;
};

template <typename T , class Context>
class MultiboxLossGradientOp final : public Operator<Context>{
public:
	MultiboxLossGradientOp(const OperatorDef& operator_def,Workspace* ws)
		:Operator<Context>(operator_def,ws),
		 num_classes_(OperatorBase::GetSingleArgument<int>("num_classes",21)),
		 share_location_(OperatorBase::GetSingleArgument<bool>("share_location",true)),
		 scale_(OperatorBase::GetSingleArgument<float>("scale",1.0)),
		 background_label_id_(OperatorBase::GetSingleArgument<int>("background_label_id",0))
		{
			loc_classes_ = share_location_ ? 1 : num_classes_;
			loc_loss_type_ = multibox_loss_param_.loc_loss_type();
			conf_loss_type_ = multibox_loss_param_.conf_loss_type();
			do_neg_mining_ = true;
		}

	USE_OPERATOR_CONTEXT_FUNCTIONS;

	bool RunOnDevice() override;
protected:
	float loc_weight_;
	int num_priors_;
	int loc_classes_;
	int num_gt_;
	int num_;
	int num_classes_;
	int background_label_id_;
	bool share_location_;
	bool use_difficult_gt_;
	bool do_neg_mining_;
	int num_matches_ ;
	int num_conf_;
//	vector<map<int, vector<int> > > all_match_indices_;
//	vector<vector<int> > all_neg_indices_;

	MultiBoxLossParameter multibox_loss_param_;

	ConfLossType conf_loss_type_;
	LocLossType loc_loss_type_;

	Tensor<Context> loc_pred_;
	Tensor<Context> loc_gt_;


	float scale_;
	Tensor<Context> conf_pred_;
	Tensor<Context> conf_gt_;
	Tensor<Context> conf_loss_;
	Tensor<Context> rowmax_;
	Tensor<Context> loss_;
	Tensor<Context> sum_multiplier_;
	Tensor<Context> prob;
	vector<map<int,vector<int> > > all_match_indices_;
	vector<vector<int> > all_neg_indices_;
};


template <>
bool MultiboxLossOp<float,CPUContext>::RunOnDevice(){
	LOG(INFO)<<"multibox forward ok1";
	const auto& loc = Input(0);
	const auto& conf = Input(1);
	const auto& prior = OperatorBase::Input<TensorCPU>(2);
	const auto& gt = OperatorBase::Input<TensorCPU>(3);

	const float* loc_data = loc.template data<float>();
	const float* conf_data = conf.template data<float>();
	const float* prior_data = prior.template data<float>();
	const float* gt_data = gt.template data<float>();

	auto* loc_pred = Output(0);
	auto* loc_gt = Output(1);
	auto* conf_pred = Output(2);
	auto* conf_gt = Output(3);
	

	num_ = loc.dim(0);
	num_priors_ = prior.dim(2) / 4;
	num_gt_ = gt.dim(2);

	CAFFE_ENFORCE_GE(num_classes_, 1, "Must provide num_classes and num_classes should not be less than 1.");

	map<int,vector<NormalizedBBox> > all_gt_bboxes;
	GetGroundTruth(gt_data,num_gt_,background_label_id_,use_difficult_gt_,&all_gt_bboxes);
    
    // item_id -> bbox(num_gt)
	

	vector<NormalizedBBox> prior_bboxes;
	vector<vector<float> > prior_variances;
	GetPriorBBoxes(prior_data,num_priors_,&prior_bboxes,&prior_variances);
	
	vector<LabelBBox> all_loc_preds;

	GetLocPredictions(loc_data,num_,num_priors_,loc_classes_,share_location_,
		&all_loc_preds);
	


	//LabelBBox label(-1) -> 8732bbox

	vector<map<int ,vector<float> > > all_match_overlaps;
	FindMatches(all_loc_preds,all_gt_bboxes,prior_bboxes,prior_variances,
		multibox_loss_param_,&all_match_overlaps,&all_match_indices_);

	// gtbox < match > priorbox
	// map<int,vector<float> >  first = label second vector prior idx - > overlaps
	// map<int,vector<int> >    first = label second vector prior idx - > gt_idx


	num_matches_ = 0;
	int num_negs = 0;

	MineHardExamples(conf,all_loc_preds,all_gt_bboxes,prior_bboxes,
	 	prior_variances,all_match_overlaps,multibox_loss_param_,
	 	&num_matches_,&num_negs,&all_match_indices_,&all_neg_indices_);


//	loc_loss_.Resize(TIndex(1));
//	conf_loss_.Resize(TIndex(1));


	/*
	if(num_matches_ >= 1){
		loc_pred_.Resize(TIndex(1),TIndex(num_matches_ * 4));
		loc_gt_.Resize(TIndex(1),TIndex(num_matches_ * 4));
		float* loc_pred_data = loc_pred_.mutable_data<float>();
		float* loc_gt_data = loc_gt_.mutable_data<float>();
		EncodeLocPrediction(all_loc_preds,all_gt_bboxes,all_match_indices_,
			prior_bboxes,prior_variances,multibox_loss_param_,
			loc_pred_data,loc_gt_data);

		// for(int k = 0; k < num_matches_ * 4; ++k){
		// 	LOG(INFO)<<"loc_gt_data = "<<loc_gt_data[k];
		// 	LOG(INFO)<<"loc_pred_data = "<<loc_pred_data[k];
		// }

		diff_.ResizeLike(loc_pred_);
		errors_.ResizeLike(loc_pred_);
		math::Sub<float,CPUContext>(num_matches_ * 4, loc_pred_data,
		 loc_gt_data, diff_.template mutable_data<float>(),&context_);

		const float* diff_data = diff_.data<float>();
		float* errors_data = errors_.mutable_data<float>();

		for(int i = 0; i < num_matches_ * 4; ++i){
			float val = diff_data[i];
			float abs_val = fabs(val);
			if(abs_val < 1.0){
				errors_data[i] = 0.5 * val * val;
			}else{
				errors_data[i] = abs_val - 0.5; 
			}
		}

		float asum = 0.;
		math::Sum<float,CPUContext>(num_matches_ * 4 ,errors_.data<float>(),&asum,&context_);

		loc_loss_.mutable_data<float>()[0] = asum;

	}else{
		loc_loss_.mutable_data<float>()[0] = 0.0;
	}
	*/
	
	if(num_matches_ >= 1){
		loc_pred->Resize(TIndex(1),TIndex(num_matches_ * 4));
		loc_gt->Resize(TIndex(1),TIndex(num_matches_ * 4));
		float* loc_pred_data = loc_pred->template mutable_data<float>();
		float* loc_gt_data = loc_gt->template mutable_data<float>();		
		EncodeLocPrediction(all_loc_preds,all_gt_bboxes,all_match_indices_,
			prior_bboxes,prior_variances,multibox_loss_param_,
			loc_pred_data,loc_gt_data);
	//	LOG(INFO)<<"loc_pred_data = "<<loc_pred_data[0];
	//	LOG(INFO)<<"loc_gt_data = "<<loc_gt_data[0];
	}else{
		loc_pred->Resize(TIndex(1),TIndex(4));
		loc_gt->Resize(TIndex(1),TIndex(4));

	}


	if(do_neg_mining_){
		num_conf_ = num_matches_ + num_negs;
	}else{
		num_conf_ = num_ * num_priors_;
	}

	LOG(INFO)<<"forward num_conf_ = "<<num_conf_;
	LOG(INFO)<<"forward num_matches_ = "<<num_matches_;

	/*
	if(num_conf_ >= 1){
		// LOG(INFO) <<"conf_loss_type_ = "<<conf_loss_type_
		// <<", MultiBoxLossParameter_ConfLossType_SOFTMAX = "
		// <<caffe::MultiBoxLossParameter_ConfLossType_SOFTMAX;
		if(conf_loss_type_ == caffe::MultiBoxLossParameter_ConfLossType_SOFTMAX){
			conf_gt_.Resize(TIndex(num_conf_));
			conf_pred_.Resize(TIndex(num_conf_),TIndex(num_classes_));
		}else{
			LOG(INFO)<<"Unknown confidence loss type.";
		}
		float* conf_pred_data = conf_pred_.mutable_data<float>();
		float* conf_gt_data = conf_gt_.mutable_data<float>();
		int count = conf_gt_.size_to_dim(0);
		math::Set<float,CPUContext>(count,float(background_label_id_),conf_gt_data,
			&context_);
		EncodeConfPrediction(conf_data,num_,num_priors_,multibox_loss_param_,
			all_match_indices_,all_neg_indices_,all_gt_bboxes,
			conf_pred_data,conf_gt_data);
		
		  for(int k = 0; k < num_conf_*num_classes_ ; ++k){
		  //	LOG(INFO)<<"conf_gt_data = "<<conf_gt_data[k];
		  	LOG(INFO)<<"conf_pred_data = "<<conf_pred_data[k];
		  }
		
		if(sum_multiplier_.size() != num_classes_){
			sum_multiplier_.Resize(num_classes_);
			math::Set<float,CPUContext>(
				num_classes_,1.f,sum_multiplier_.mutable_data<float>(),&context_);
		}
		prob.ResizeLike(conf_pred_);
		float* Pdata = prob.mutable_data<float>();
		rowmax_.Resize(num_conf_);
		loss_.Resize(num_conf_);

		SoftmaxCPU(
			context_,
			num_conf_,
			num_classes_,
			conf_pred_data,
			Pdata,
			loss_.mutable_data<float>(),
			sum_multiplier_.mutable_data<float>(),
			1,
			rowmax_.mutable_data<float>()
			);
		float loss_sum = 0;
		for(int k = 0; k < num_conf_; ++k ){
			CAFFE_ENFORCE(conf_gt_data[k]<num_classes_&&conf_gt_data[k]>=0);
			float l = -Pdata[ k * num_classes_ + static_cast<int>(conf_gt_data[k]) ];
			loss_sum += l ;
		}
	//	math::Exp(num_conf_ * num_classes_,Pdata,Pdata,&context_);
		
		float* conf_loss_data = conf_loss_.mutable_data<float>();
		conf_loss_data[0] = loss_sum * scale_;
	}else{
		conf_loss_.mutable_data<float>()[0] = 0.0;
	}
	*/
	
	if(num_conf_>=1){
		conf_gt->Resize(TIndex(num_conf_));
		conf_pred->Resize(TIndex(num_conf_),TIndex(num_classes_));
		float* conf_pred_data = conf_pred->template mutable_data<float>();
		float* conf_gt_data = conf_gt->template mutable_data<float>();
		int count = conf_gt->size_to_dim(0);
		math::Set<float,CPUContext>(count,float(background_label_id_),conf_gt_data,
			&context_);
		EncodeConfPrediction(conf_data,num_,num_priors_,multibox_loss_param_,
			all_match_indices_,all_neg_indices_,all_gt_bboxes,
			conf_pred_data,conf_gt_data);
	}else{
		conf_gt->Resize(TIndex(0));
		conf_pred->Resize(TIndex(0),TIndex(num_classes_));
		
	}
	
	
//	LOG(INFO)<<"loc_loss = "<<loc_loss_.data<float>()[0];
//	LOG(INFO)<<"conf_loss = "<<conf_loss_.data<float>()[0];
	all_match_indices_.clear();
	all_neg_indices_.clear();

	return true;
}

template <>
bool MultiboxLossGradientOp<float,CPUContext>::RunOnDevice(){
	const auto& loc = Input(0);
	const auto& conf = Input(1);
	const auto& prior = OperatorBase::Input<TensorCPU >(2);
	const auto& gt = OperatorBase::Input<TensorCPU >(3);
	const auto& loc_diff = Input(4);
	const auto& conf_diff = Input(5); 

	auto* dloc = Output(0);
	auto* dconf = Output(1);
	dloc->ResizeLike(loc);
	dconf->ResizeLike(conf);

	const float* loc_data = loc.template data<float>();
	const float* conf_data = conf.template data<float>();
	const float* prior_data = prior.template data<float>();
	const float* gt_data = gt.template data<float>();

	num_ = loc.dim(0);
	num_priors_ = prior.dim(2)/4;
	num_gt_ = gt.dim(2);
	int dim_loc = loc.size() / num_;
	int dim_conf = conf.size() / num_;
	CAFFE_ENFORCE_GE(num_classes_,1,"Must provide num_classes and num_classes should not be less than 1.");

	map<int,vector<NormalizedBBox> > all_gt_bboxes;
	GetGroundTruth(gt_data,num_gt_,background_label_id_,use_difficult_gt_,&all_gt_bboxes);
    
    // item_id -> bbox(num_gt)
	//LOG(INFO)<<"loc dims = "<<loc.dims()<<" conf dims = "<<conf.dims()<<"prior dims = "<<prior.dims();
    //    LOG(INFO)<<"gt dims = "<<gt.dims();
	vector<NormalizedBBox> prior_bboxes;
	vector<vector<float> > prior_variances;
	GetPriorBBoxes(prior_data,num_priors_,&prior_bboxes,&prior_variances);

	vector<LabelBBox> all_loc_preds;

	GetLocPredictions(loc_data,num_,num_priors_,loc_classes_,share_location_,
		&all_loc_preds);

	//LabelBBox label(-1) -> 8732bbox

	vector<map<int ,vector<float> > > all_match_overlaps;
	FindMatches(all_loc_preds,all_gt_bboxes,prior_bboxes,prior_variances,
		multibox_loss_param_,&all_match_overlaps,&all_match_indices_);

	// gtbox < match > priorbox
	// map<int,vector<float> >  first = label second vector prior idx - > overlaps
	// map<int,vector<int> >    first = label second vector prior idx - > gt_idx


	num_matches_ = 0;
	int num_negs = 0;

	MineHardExamples(conf,all_loc_preds,all_gt_bboxes,prior_bboxes,
	 	prior_variances,all_match_overlaps,multibox_loss_param_,
	 	&num_matches_,&num_negs,&all_match_indices_,&all_neg_indices_);
	const float normalizer = 1./num_matches_;
	float* dloc_data = dloc->template mutable_data<float>();
	math::Set<float, CPUContext>(dloc->size(),0.,dloc_data,&context_);

	if(do_neg_mining_){
		num_conf_ = num_matches_ + num_negs;
	}else{
		num_conf_ = num_ * num_priors_;
	}
	LOG(INFO)<<"backward num_conf_ = "<<num_conf_;
	LOG(INFO)<<"backward num_matches_ = "<<num_matches_;

	if(num_matches_ >= 1){
		loc_pred_.ResizeLike(loc_diff);
		float* loc_pred_data = loc_pred_.template mutable_data<float>();
//		const float* loc_diff_data = loc_diff.data<float>()
		int loc_count = loc_pred_.size();
		math::Scale<float, CPUContext>(loc_count,
				normalizer * num_matches_ ,loc_diff.template data<float>(),loc_pred_data,&context_);
		int count = 0;
		for(int i = 0; i < num_; ++i){
			for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {
        	  const int label = share_location_ ? 0 : it->first;
         	 const vector<int>& match_index = it->second;
         	 for (int j = 0; j < match_index.size(); ++j) {
            	if (match_index[j] <= -1) {
              		continue;
            	}
            // Copy the diff to the right place.
            	int start_idx = loc_classes_ * 4 * j + label * 4;
            //	caffe_copy<Dtype>(4, loc_pred_diff + count * 4,
            //    	              loc_bottom_diff + start_idx);
            	memcpy(dloc_data+start_idx, loc_pred_data + count * 4, 4 * sizeof(float));
            	++count;
          		}
       	 	}
        	dloc_data += dim_loc;
      	}
	}
	
	conf_pred_.ResizeLike(conf);
	float* conf_pred_diff = conf_pred_.template mutable_data<float>();
	float* dconf_data = dconf->template mutable_data<float>();
	math::Set<float, CPUContext>(dconf->size(), 0., dconf_data, &context_);
	if(num_conf_ >= 1){
		math::Scale<float, CPUContext>(conf_diff.size(),
			normalizer * num_conf_ ,conf_diff.template data<float>(),
			conf_pred_diff, &context_);
 		
		if(do_neg_mining_){
			int count = 0;
			for(int i = 0; i < num_; ++i){
				
				const map<int, vector<int> >& match_indices = all_match_indices_[i];
	          for (map<int, vector<int> >::const_iterator it =
	               match_indices.begin(); it != match_indices.end(); ++it) {
	            const vector<int>& match_index = it->second;
	        	
	            CAFFE_ENFORCE_EQ(match_index.size(), num_priors_);
	            for (int j = 0; j < num_priors_; ++j) {
	              if (match_index[j] <= -1) {
	                continue;
	              }
	              
	              memcpy(dconf_data + j * num_classes_,
	              	conf_pred_diff + count * num_classes_,
	              	num_classes_ * sizeof(float));
	              
	              ++count;
	            }
	          }
	          
	          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
	            int j = all_neg_indices_[i][n];
	            CAFFE_ENFORCE_LT(j, num_priors_);
	            memcpy(dconf_data + j * num_classes_,
	            	conf_pred_diff + count * num_classes_,
	            	num_classes_ * sizeof(float));
	            ++count;
	          }
	          
	          dconf_data += dim_conf;
			}
		}
	}
	all_match_indices_.clear();
	all_neg_indices_.clear();

	LOG(INFO)<<"mulitbox backward ok";
	return true;
}
/*
template <>
bool MultiboxLossOp<float,CUDAContext>::RunOnDevice(){
	const auto& loc = Input(0);
	const auto& conf = Input(1);
	const auto& prior = OperatorBase::Input<TensorCPU>(2);
	const auto& gt = OperatorBase::Input<TensorCPU>(3);

	const float* loc_data = loc.template data<float>();
	const float* conf_data = conf.template data<float>();
	const float* prior_data = prior.template data<float>();
	const float* gt_data = gt.template data<float>();

	auto* loc_pred = Output(0);
	auto* loc_gt = Output(1);
	auto* conf_pred = Output(2);
	auto* conf_gt = Output(3);
	

	num_ = loc.dim(0);
	num_priors_ = prior.dim(2) / 4;
	num_gt_ = gt.dim(2);

	CAFFE_ENFORCE_GE(num_classes_, 1, "Must provide num_classes and num_classes should not be less than 1.");

	map<int,vector<NormalizedBBox> > all_gt_bboxes;
	GetGroundTruth(gt_data,num_gt_,background_label_id_,use_difficult_gt_,&all_gt_bboxes);


	vector<NormalizedBBox> prior_bboxes;
	vector<vector<float> > prior_variances;
	GetPriorBBoxes(prior_data,num_priors_,&prior_bboxes,&prior_variances);
	
	vector<LabelBBox> all_loc_preds;

	GetLocPredictions(loc_data,num_,num_priors_,loc_classes_,share_location_,
		&all_loc_preds);
	

	vector<map<int ,vector<float> > > all_match_overlaps;
	FindMatches(all_loc_preds,all_gt_bboxes,prior_bboxes,prior_variances,
		multibox_loss_param_,&all_match_overlaps,&all_match_indices_);


	num_matches_ = 0;
	int num_negs = 0;

	MineHardExamples(conf,all_loc_preds,all_gt_bboxes,prior_bboxes,
	 	prior_variances,all_match_overlaps,multibox_loss_param_,
	 	&num_matches_,&num_negs,&all_match_indices_,&all_neg_indices_);


	if(num_matches_ >= 1){
		loc_pred->Resize(TIndex(1),TIndex(num_matches_ * 4));
		loc_gt->Resize(TIndex(1),TIndex(num_matches_ * 4));
		float* loc_pred_data = loc_pred->template mutable_data<float>();
		float* loc_gt_data = loc_gt->template mutable_data<float>();		
		EncodeLocPrediction(all_loc_preds,all_gt_bboxes,all_match_indices_,
			prior_bboxes,prior_variances,multibox_loss_param_,
			loc_pred_data,loc_gt_data);

	}else{
		loc_pred->Resize(TIndex(1),TIndex(0));
		loc_gt->Resize(TIndex(1),TIndex(0));
	}


	if(do_neg_mining_){
		num_conf_ = num_matches_ + num_negs;
	}else{
		num_conf_ = num_ * num_priors_;
	}


	if(num_conf_>=1){
		conf_gt->Resize(TIndex(num_conf_));
		conf_pred->Resize(TIndex(num_conf_),TIndex(num_classes_));
		float* conf_pred_data = conf_pred->template mutable_data<float>();
		float* conf_gt_data = conf_gt->template mutable_data<float>();
		int count = conf_gt->size_to_dim(0);
		math::Set<float,CUDAContext>(count,float(background_label_id_),conf_gt_data,
			&context_);
		EncodeConfPrediction(conf_data,num_,num_priors_,multibox_loss_param_,
			all_match_indices_,all_neg_indices_,all_gt_bboxes,
			conf_pred_data,conf_gt_data);
	}else{
		conf_gt->Resize(TIndex(0));
		conf_pred->Resize(TIndex(0),TIndex(num_classes_));
		
	}

	all_match_indices_.clear();
	all_neg_indices_.clear();

	return true;
}

template <>
bool MultiboxLossGradientOp<float,CUDAContext>::RunOnDevice(){
	const auto& loc = Input(0);
	const auto& conf = Input(1);
	const auto& prior = OperatorBase::Input<TensorCPU >(2);
	const auto& gt = OperatorBase::Input<TensorCPU >(3);
	const auto& loc_diff = Input(4);
	const auto& conf_diff = Input(5); 

	auto* dloc = Output(0);
	auto* dconf = Output(1);
	dloc->ResizeLike(loc);
	dconf->ResizeLike(conf);

	const float* loc_data = loc.template data<float>();
	const float* conf_data = conf.template data<float>();
	const float* prior_data = prior.template data<float>();
	const float* gt_data = gt.template data<float>();

	num_ = loc.dim(0);
	num_priors_ = prior.dim(2)/4;
	num_gt_ = gt.dim(2);
	int dim_loc = loc.size() / num_;
	int dim_conf = conf.size() / num_;
	CAFFE_ENFORCE_GE(num_classes_,1,"Must provide num_classes and num_classes should not be less than 1.");

	map<int,vector<NormalizedBBox> > all_gt_bboxes;
	GetGroundTruth(gt_data,num_gt_,background_label_id_,use_difficult_gt_,&all_gt_bboxes);
    
 
	vector<NormalizedBBox> prior_bboxes;
	vector<vector<float> > prior_variances;
	GetPriorBBoxes(prior_data,num_priors_,&prior_bboxes,&prior_variances);

	vector<LabelBBox> all_loc_preds;

	GetLocPredictions(loc_data,num_,num_priors_,loc_classes_,share_location_,
		&all_loc_preds);

	//LabelBBox label(-1) -> 8732bbox

	vector<map<int ,vector<float> > > all_match_overlaps;
	FindMatches(all_loc_preds,all_gt_bboxes,prior_bboxes,prior_variances,
		multibox_loss_param_,&all_match_overlaps,&all_match_indices_);


	num_matches_ = 0;
	int num_negs = 0;

	MineHardExamples(conf,all_loc_preds,all_gt_bboxes,prior_bboxes,
	 	prior_variances,all_match_overlaps,multibox_loss_param_,
	 	&num_matches_,&num_negs,&all_match_indices_,&all_neg_indices_);
	const float normalizer = 1./num_matches_;
	float* dloc_data = dloc->template mutable_data<float>();
	math::Set<float, CUDAContext>(dloc->size(),0.,dloc_data,&context_);
	if(do_neg_mining_){
		num_conf_ = num_matches_ + num_negs;
	}else{
		num_conf_ = num_ * num_priors_;
	}
	if(num_matches_ >= 1){
		loc_pred_.ResizeLike(loc_diff);
		float* loc_pred_data = loc_pred_.template mutable_data<float>();

		int loc_count = loc_pred_.size();
		math::Scale<float, CUDAContext>(loc_count,
				normalizer * num_matches_ ,loc_diff.template data<float>(),loc_pred_data,&context_);
		int count = 0;
		for(int i = 0; i < num_; ++i){
			for (map<int, vector<int> >::iterator it =
             all_match_indices_[i].begin();
             it != all_match_indices_[i].end(); ++it) {
        	  const int label = share_location_ ? 0 : it->first;
         	 const vector<int>& match_index = it->second;
         	 for (int j = 0; j < match_index.size(); ++j) {
            	if (match_index[j] <= -1) {
              		continue;
            	}
            // Copy the diff to the right place.
            	int start_idx = loc_classes_ * 4 * j + label * 4;

            	memcpy(dloc_data+start_idx, loc_pred_data + count * 4, 4 * sizeof(float));
            	++count;
          		}
       	 	}
        	dloc_data += dim_loc;
      	}
	}
	
	conf_pred_.ResizeLike(conf);
	float* conf_pred_diff = conf_pred_.template mutable_data<float>();
	float* dconf_data = dconf->template mutable_data<float>();
	math::Set<float, CUDAContext>(dconf->size(), 0., dconf_data, &context_);
	if(num_conf_ >= 1){
		math::Scale<float, CUDAContext>(conf_diff.size(),
			normalizer * num_conf_ ,conf_diff.template data<float>(),
			conf_pred_diff, &context_);
 		
		if(do_neg_mining_){
			int count = 0;
			for(int i = 0; i < num_; ++i){
				
				const map<int, vector<int> >& match_indices = all_match_indices_[i];
	          for (map<int, vector<int> >::const_iterator it =
	               match_indices.begin(); it != match_indices.end(); ++it) {
	            const vector<int>& match_index = it->second;
	        	
	            CAFFE_ENFORCE_EQ(match_index.size(), num_priors_);
	            for (int j = 0; j < num_priors_; ++j) {
	              if (match_index[j] <= -1) {
	                continue;
	              }
	              
	              memcpy(dconf_data + j * num_classes_,
	              	conf_pred_diff + count * num_classes_,
	              	num_classes_ * sizeof(float));
	              
	              ++count;
	            }
	          }
	          
	          for (int n = 0; n < all_neg_indices_[i].size(); ++n) {
	            int j = all_neg_indices_[i][n];
	            CAFFE_ENFORCE_LT(j, num_priors_);
	            memcpy(dconf_data + j * num_classes_,
	            	conf_pred_diff + count * num_classes_,
	            	num_classes_ * sizeof(float));
	            ++count;
	          }
	          
	          dconf_data += dim_conf;
			}
		}
	}
	all_match_indices_.clear();
	all_neg_indices_.clear();

	
	return true;
}
*/

}


#endif
