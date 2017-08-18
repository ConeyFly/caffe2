#include "caffe2/operators/norm_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2{

template <typename T>
__global__ void DivBsx(const int nthreads, const T* A,
    const T* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    T* B) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] / v[c];
    } else {
      B[index] = A[index] / v[r];
    }
  }
}

template <typename T>
__global__ void MulBsx(const int nthreads, const T* A,
    const T* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
   	T* B) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}


__global__ void Div(const int N, const float* a, const float* b, float* y)
{
   CUDA_1D_KERNEL_LOOP(i, N) {  
    y[i] = a[i] / b[i];   
  }
}

__global__ void Add(const int N, const float* a, const float* b, float* y)
{
   CUDA_1D_KERNEL_LOOP(i, N) {  
    y[i] = a[i] + b[i];   
  }
}


template <>
bool NormOp<float,CUDAContext>::RunOnDevice(){
	
	const auto& X = Input(0);
	const auto& scale = Input(1);
	auto* Y =  Output(0);
	auto* norm = Output(1);
	int num = X.dim(0);
	int channels = X.dim(1);
	int height = X.dim(2);
	int width = X.dim(3);

	Y->ResizeLike(X);
	buffer_.Resize(1,channels,height,width);
	buffer_channel_.Resize(1,channels,1,1);
	buffer_spatial_.Resize(1,1,height,width);

	if(across_spatial_){
		norm->Resize(num, 1, 1, 1);
	}else{
		norm->Resize(num, 1, height, width);
	}
	int spatial_dim = height*width;
	sum_channel_multiplier_.Resize(1, channels, 1, 1);
	math::Set<float,CUDAContext>(channels, 
			float(1.0), sum_channel_multiplier_.mutable_data<float>(),
			&context_);
	sum_spatial_multiplier_.Resize(1,1,height,width);
	math::Set<float,CUDAContext>(spatial_dim,
			float(1.0), sum_spatial_multiplier_.mutable_data<float>(),
			&context_);
	
	if(channels_shared_){
		CAFFE_ENFORCE_EQ(scale.dim(0),1);
	}else{
		CAFFE_ENFORCE_EQ(scale.dim(0),channels);
	}

	const float* Xdata = X.data<float>();
	const float* Sdata = scale.data<float>();
	
	
	
	float* buffer_data = buffer_.mutable_data<float>();
	float* norm_data = norm->mutable_data<float>();

	math::Set<float,CUDAContext>(norm->size(),float(eps_),norm_data,&context_);
	const float* sum_channel_multiplier = sum_channel_multiplier_.data<float>();
	const float* sum_spatial_multiplier = sum_spatial_multiplier_.data<float>();
	float* Ydata = Y->mutable_data<float>();
	int dim = channels * height * width;	

	for(int n = 0; n < num; ++n){
		math::Powx<float, CUDAContext>(dim, Xdata, float(2.0), buffer_data, &context_);
		if(across_spatial_){
			float gpu_sum ;
			math::Sum<float, CUDAContext>(dim, buffer_data, &gpu_sum, &context_);
			norm_data[n] = pow(gpu_sum + eps_,float(0.5));
			math::Scale<float,CUDAContext>(dim, float(1.0 / norm_data[n]), Xdata, Ydata,
					&context_);
		}else{
			math::Gemv<float, CUDAContext>(CblasTrans, channels, spatial_dim, 
					    float(1.0), buffer_data, sum_channel_multiplier,float(1.0),
					    norm_data,&context_);
			math::Powx<float, CUDAContext>(spatial_dim, norm_data, float(0.5), 
								norm_data, &context_);
			DivBsx<float><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS, 0, 
				context_.cuda_stream()>>>(
				  dim, Xdata, norm_data, channels, spatial_dim,CblasNoTrans, Ydata);
			norm_data += spatial_dim;
		}
		if(channels_shared_){
			math::Scale<float, CUDAContext>(dim, Sdata[0], Ydata, Ydata, &context_);
		} else { 
			MulBsx<float><<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS, 0, 
			    context_.cuda_stream()>>>(
				dim, Ydata, Sdata, channels, spatial_dim, CblasTrans, Ydata);
		}
		Xdata += dim;
		Ydata += dim;
	}
	
	return true;
}

template <>
bool NormGradientOp<float,CUDAContext>::RunOnDevice()
{
	
	auto& X = Input(0); // bottom_data
	auto& scale = Input(1);//scale
	auto& Y = Input(2); // top_data
	auto& dY = Input(3); //top_diff
	auto& norm = Input(4);
	auto* dX = Output(0); // bottom_diff
	auto* dscale = Output(1); // scale_diff

	const int count = Y.size();
	const int num = Y.dim(0);
	const int dim = count / num ;
	const int spatial_dim = Y.dim(2) * Y.dim(3);
	const int channels = Y.dim(1);
	const int height = Y.dim(2);
	const int width = Y.dim(3);
	dX->ResizeLike(X);
	dscale->ResizeLike(scale);
	buffer_.Resize(1,channels,height,width);
	buffer_channel_.Resize(1,channels,1,1);
	buffer_spatial_.Resize(1,1,height,width);
	sum_channel_multiplier_.Resize(1,channels,1,1);
	sum_spatial_multiplier_.Resize(1,1,height,width);
	


	const float* Xdata = X.data<float>();
	float* dXdata = dX->mutable_data<float>();
	const float* Ydata = Y.data<float>();	
	const float* dYdata = dY.data<float>();
	
	const float* Sdata = scale.data<float>();
	float* dscaledata = dscale->mutable_data<float>();
	float* buffer_data = buffer_.mutable_data<float>();
	float* buffer_channel = buffer_channel_.mutable_data<float>();
	float* buffer_spatial = buffer_spatial_.mutable_data<float>();
	
	sum_channel_multiplier_.Resize(1,channels,1,1);
	float* sum_channel_multiplier = sum_channel_multiplier_.mutable_data<float>();
	math::Set<float,CUDAContext>(channels,1.0,sum_channel_multiplier,&context_);
	
	sum_spatial_multiplier_.Resize(1,1,height,width);
	
	float* sum_spatial_multiplier = sum_spatial_multiplier_.mutable_data<float>();
	math::Set<float,CUDAContext>(spatial_dim,1.0,sum_spatial_multiplier,&context_);
	
	const float* norm_data = norm.data<float>();
	
	if(channels_shared_){
		float td;
		math::Dot<float,CUDAContext>(count, Ydata ,dYdata, &td,&context_);
		dscaledata[0] += td / Sdata[0];
	}else{
		math::Set<float, CUDAContext>(channels, 0.0, dscaledata,&context_);
		for(int n = 0; n < num; ++n){
		//	LOG(INFO)<<"ok1";
			math::Mul<float,CUDAContext>(dim, Ydata + n * dim, dYdata + n * dim , buffer_data ,&context_);
			math::Gemv<float,CUDAContext>(CblasNoTrans, channels, spatial_dim, float(1.0),
				buffer_data, sum_spatial_multiplier, float(0.0), buffer_channel, &context_);
			math::Div<float,CUDAContext>(channels, buffer_channel, Sdata, 
				buffer_channel, &context_);
		//	Div<<<CAFFE_GET_BLOCKS(channels), CAFFE_CUDA_NUM_THREADS, 0, 
		//		context_.cuda_stream()>>>(channels,buffer_channel, Sdata, buffer_channel);
 		//	printf("%lf\n",buffer_channel[0]);
			math::Add<float,CUDAContext>(channels, buffer_channel, dscaledata,
				dscaledata ,&context_);
		//	Add<<<CAFFE_GET_BLOCKS(channels), CAFFE_CUDA_NUM_THREADS, 0, 
		//		context_.cuda_stream()>>>(channels,buffer_channel, dscaledata, dscaledata);
		}
		
	}
	for(int n = 0; n < num; ++n){
		if(across_spatial_){
			float a;
			math::Dot<float, CUDAContext>(dim, Xdata, dYdata, &a, &context_);
			math::Scale<float, CUDAContext>(dim, a / norm_data[n] / norm_data[n], 
				Xdata, dXdata, &context_);
			math::Sub<float, CUDAContext>(dim, dYdata, dXdata, dXdata, &context_);
			math::Scale<float, CUDAContext>(dim, 1.0 / norm_data[n], dXdata, dXdata, &context_);
		}else{
			math::Mul<float, CUDAContext>(dim, Xdata, dYdata, buffer_data, &context_);
			math::Gemv<float, CUDAContext>(CblasTrans, channels, spatial_dim, 1.0,
				buffer_data, sum_channel_multiplier, 0.,
				buffer_spatial, &context_);
			MulBsx<float><<<CAFFE_GET_BLOCKS(dim),
			CAFFE_CUDA_NUM_THREADS,
			0,
			context_.cuda_stream()
			>>>(dim, Xdata, buffer_spatial, channels,
				spatial_dim, CblasNoTrans, dXdata);
		
			math::Powx<float, CUDAContext>(spatial_dim, norm_data, 2.,
				buffer_spatial, &context_);
			DivBsx<<<CAFFE_GET_BLOCKS(dim),
			CAFFE_CUDA_NUM_THREADS,
			0,
			context_.cuda_stream()
			>>>(dim, dXdata, buffer_spatial, channels, spatial_dim,
				CblasNoTrans, dXdata);
			math::Sub<float, CUDAContext>(dim, dYdata, dXdata, dXdata,&context_);
			DivBsx<<<CAFFE_GET_BLOCKS(dim),
			CAFFE_CUDA_NUM_THREADS,
			0,
			context_.cuda_stream()
			>>>(dim, dXdata, norm_data, channels,spatial_dim,
				CblasNoTrans,dXdata);
			norm_data += spatial_dim;
		}

		if(channels_shared_){
			math::Scale<float, CUDAContext>(dim, Sdata[0], dXdata, dXdata, &context_);
		}else{
			MulBsx<<<CAFFE_GET_BLOCKS(dim),
			CAFFE_CUDA_NUM_THREADS,
			0,
			context_.cuda_stream()
			>>>(dim, dXdata, Sdata, channels, spatial_dim,
				CblasTrans, dXdata);
		}
		Xdata += dim;
		dYdata += dim;
		dXdata += dim;
	}
	
	
	return true;
}

namespace{
REGISTER_CUDA_OPERATOR(Norm,NormOp<float,CUDAContext>);
REGISTER_CUDA_OPERATOR(NormGradient,NormGradientOp<float,CUDAContext>);

}

}
