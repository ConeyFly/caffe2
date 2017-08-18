#ifndef CAFFE2_OPERATORS_DETECTION_EVALUTE_OP_H
#define CAFFE2_OPERATORS_DETECTION_EVALUTE_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe2/utils/bbox_util.h"
#include "caffe2/utils/image_util.h"

namespace caffe2{

template <typename T, class Context>
class DetectionEvaluteOp final : public Operator<Context>{
public:
	DetectionEvaluteOp(const OperatorDef& operator_def, Workspace* ws)
		:Operator<Context>(operator_def, ws),
		num_classes_(OperatorBase::GetSingleArgument<int>("num_classes",-1)),
		background_label_id_(OperatorBase::GetSingleArgument<int>("background_label_id",0)),
		overlap_threshold_(OperatorBase::GetSingleArgument<float>("overlap_threshold",0.5)),
		evaluate_difficult_gt_(OperatorBase::GetSingleArgument<bool>("evaluate_difficult_gt",false))
		{
			CAFFE_ENFORCE_NE(num_classes_,-1,"Must provide num_classes");
			CAFFE_ENFORCE_GT(overlap_threshold_,0.,"overlap_threshold must be non negative");
			LOG(INFO)<<"name_size_file";
			
			if(OperatorBase::HasArgument("name_size_file")){
				LOG(INFO)<<"name_size_file";
				name_size_file_ = OperatorBase::GetSingleArgument<string>("name_size_file","");
				std::ifstream infile(name_size_file_.c_str());
				CAFFE_ENFORCE(infile.good(),"Failed to open name size file!");
				string name;
				int height, width;
				while(infile >> name >> height >> width){
					sizes_.push_back(std::make_pair(height, width));
				}
				infile.close();
			}
			count_ = 0;
			use_normalized_bbox_ = sizes_.size() == 0;
			
			resize_param_.resize_valid = OperatorBase::template GetSingleArgument<bool>("resize_valid",true);
			resize_param_.resize_mode = OperatorBase::template GetSingleArgument<string>("resize_mode","WARP");
			resize_param_.height = OperatorBase::template GetSingleArgument<int>("resize_height",300);
    			resize_param_.width = OperatorBase::template GetSingleArgument<int>("resize_width",300);
			resize_param_.height_scale = OperatorBase::template GetSingleArgument<int>("height_scale",0);
    			resize_param_.height_scale = OperatorBase::template GetSingleArgument<int>("width_scale",0);
			has_resize_ = resize_param_.resize_valid;		
		}
		USE_OPERATOR_CONTEXT_FUNCTIONS;

	    bool RunOnDevice() override;
protected:
	int num_classes_;
	int background_label_id_;
	float overlap_threshold_;
	bool evaluate_difficult_gt_;
	string name_size_file_;
	vector<pair<int, int> >sizes_;
	int count_;
	bool use_normalized_bbox_;
	bool has_resize_;
	ResizeParam resize_param_;
};

}

#endif
