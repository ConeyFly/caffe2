#include<vector>

#include "caffe2/operators/smooth_L1_loss_op.h"
#include "caffe2/core/context_gpu.h"
//#include "caffe2/utils/math_gpu.cu";
namespace caffe2{



__global__ void SmoothL1Forward(const int n,const float* in, float* out){
	CUDA_1D_KERNEL_LOOP(i, n){
		float val = in[i];
		float abs_val = abs(val);
		if(abs_val < 1){
			out[i] = 0.5 * val * val;
		}else{
			out[i] = abs_val - 0.5;
		}
	}
}


__global__ void SmoothL1Backward(const int n,const float* in, float* out){
	CUDA_1D_KERNEL_LOOP(i, n){
		float val = in[i];
		float abs_val = abs(val);
		if(abs_val < 1.){
			out[i] = val;
		}else{
			out[i] = (val > 0.) - (val < 0.);
		}
	}
}

template<>
bool SmoothL1LossOp<float, CUDAContext>::RunOnDevice(){
    
    const auto& X1 = Input(0);
    const auto& X2 = Input(1);
    auto* Y = OperatorBase::Output<TensorCUDA>(0);
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

    math::Sub<float,CUDAContext>(count1,X1data,X2data,
    	diff_.mutable_data<float>(),&context_);

    if(has_weights_){
    	math::Mul<float,CUDAContext>(count1,weights,diff_.data<float>(),
            diff_.mutable_data<float>(),&context_); 	
    }

    SmoothL1Forward<<<
    CAFFE_GET_BLOCKS(count1),
    CAFFE_CUDA_NUM_THREADS,
    0,
    context_.cuda_stream()
    >>>(count1, diff_.data<float>(),errors_.mutable_data<float>());
    
    float alpha = 1.0 / float(X1.dim(0)) / (count1 / 4);
    math::Sum<float, CUDAContext>(count1,errors_.data<float>(),Ydata,&context_);
    math::Scale<float, CUDAContext>(1,alpha,Y->data<float>(),Ydata,&context_);
   // LOG(INFO)<<"asum = "<<asum;
   // Ydata[0] = asum / float(X1.dim(0)) / (count1 / 4);
    
    return true;
}

template<>
bool SmoothL1LossGradientOp<float, CUDAContext>::RunOnDevice(){
    
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
    math::Sub<float,CUDAContext>(count,X1data,X2data,diff_data,&context_);

    SmoothL1Backward<<<
    CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS,
    0,
    context_.cuda_stream()
    >>>(count, diff_.data<float>(), diff_.mutable_data<float>());
    const float alpha = 1.0 / X1.dim(0) / (count / 4);
  //  float alpha1;
  //  math::Div<float, CUDAContext>(1,d_avg_loss.data<float>(),alpha,alpha1,&context_);
    math::Set<float, CUDAContext>(count, 0., Y->mutable_data<float>(),&context_);
    math::Axpby<float,CUDAContext>(
                                count, 
                                alpha, 
                                diff_.mutable_data<float>(),
                                0.,
                                Y->mutable_data<float>(),
                                &context_);
    
    return true;

}


namespace{
REGISTER_CUDA_OPERATOR(SmoothL1Loss,SmoothL1LossOp<float,CUDAContext>);
REGISTER_CUDA_OPERATOR(SmoothL1LossGradient,SmoothL1LossGradientOp<float,CUDAContext>);


}

}
