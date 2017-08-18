#include "caffe2/operators/smooth_L1_loss_op.h"


namespace caffe2{

template <>
bool SmoothL1LossOp<float,CPUContext>::RunOnDevice(){
	const auto& X1 = Input(0);
	const auto& X2 = Input(1);
	auto* Y = OperatorBase::Output<TensorCPU>(0);
	const float* weights = (InputSize() > 2 ? Input(2).data<float>():nullptr);
    
    Y->Resize(vector<TIndex>());
    float* Ydata = Y->mutable_data<float>();
    
    const int count1 = X1.size();
    const int count2 = X2.size();

    CAFFE_ENFORCE_EQ(count1,count2);


    const float* X1data = X1.data<float>();
    const float* X2data = X2.data<float>();
    
    diff_.ResizeLike(X1);
    errors_.ResizeLike(X2);
    math::Sub<float,CPUContext>(count1,X1data,X2data,diff_.template mutable_data<float>(),&context_);

    if(has_weights_){
    	math::Mul<float,CPUContext>(count1,weights,diff_.data<float>(),
            diff_.template mutable_data<float>(),&context_); 	
    }

    const float* diff_data = diff_.data<float>();
    float* errors_data = errors_.mutable_data<float>();

    for(int i=0; i < count1; ++i){
    	float val = diff_data[i];
    	float abs_val = fabs(val);
     //   LOG(INFO)<<"abs_val = "<<abs_val;
    	if(abs_val < 1.){
    		errors_data[i] = 0.5 * val * val;
    	}else{
    		errors_data[i] = abs_val - 0.5;
    	}
    }

    float asum = 0.;
    math::Sum<float,CPUContext>(count1,errors_.data<float>(),&asum,&context_);
    Ydata[0] = asum / float(X1.dim(0)) / (count1 / 4);


    return true ;
}


template <>
bool SmoothL1LossGradientOp<float, CPUContext>::RunOnDevice(){
    const auto& X1 = Input(0);
    const auto& X2 = Input(1); 
    const auto& d_avg_loss = Input(InputSize() - 1);
    const float* X1data = X1.data<float>();
    const float* X2data = X2.data<float>();
    auto* Y = Output(0);
    Y->ResizeLike(X1);
    diff_.ResizeLike(X1);
    const int count = diff_.size();
    float* diff_data = diff_.mutable_data<float>();
    math::Sub<float,CPUContext>(count,X1data,X2data,diff_data,&context_);
    
    for(int i = 0; i < count; ++i){
        float val = diff_data[i];
        if(fabs(val) < 1.){
            diff_data[i] = val;
        }else{
            diff_data[i] = (val > 0.) - (val < 0.);
        }
    }
  //  LOG(INFO)<<"d_avg_loss = "<<d_avg_loss.data<float>()[0];
    const float alpha = d_avg_loss.data<float>()[0] / X1.dim(0) / (count / 4);
    math::Axpby<float,CPUContext>(
                                count, 
                                alpha, 
                                diff_.template mutable_data<float>(),
                                0.,
                                Y->template mutable_data<float>(),
                                &context_);
    return true;
}

namespace{
REGISTER_CPU_OPERATOR(SmoothL1Loss,SmoothL1LossOp<float,CPUContext>);
REGISTER_CPU_OPERATOR(SmoothL1LossGradient,SmoothL1LossGradientOp<float,CPUContext>);


OPERATOR_SCHEMA(SmoothL1Loss)
	.NumInputs(2,3)
	.NumOutputs(1)
	.TensorInferenceFunction(
		[](const OperatorDef& def,const vector<TensorShape>& in){
			ArgumentHelper helper(def);
			bool has_weights_ = helper.GetSingleArgument<bool>("has_weights",0);
			vector<TensorShape> out(1);
			auto X1 = in[0];
			auto X2 = in[1];

			out[0].set_data_type(X1.data_type());

			return out; 
		})
	.SetDoc(R"DOC()DOC")
	.Arg("has_weights","")
	.Arg("order","NCHW")
	.Input(0,"X1","")
	.Input(1,"X2","")
	.Input(2,"weights","")
	.Output(0,"smooth_L1_loss","");
}

OPERATOR_SCHEMA(SmoothL1LossGradient).NumInputs(2,3).NumOutputs(1);
class GetSmoothL1LossGradient : public GradientMakerBase{
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override {
	CHECK_EQ(def_.input_size(),2);
	return SingleGradientDef(
	    "SmoothL1LossGradient","",
	    vector<string>{I(0),I(1),GO(0)},
	    vector<string>{GI(0)}
	);
    }
};
REGISTER_GRADIENT(SmoothL1Loss,GetSmoothL1LossGradient);

}
