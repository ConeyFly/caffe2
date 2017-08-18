#include "caffe2/operators/norm_op.h"

namespace caffe2{

template <>
bool NormOp<float,CPUContext>::RunOnDevice(){
	
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
		norm->Resize(num,1,1,1);
	}else{
		norm->Resize(num,1,height,width);
	}
	int spatial_dim = height*width;
	sum_channel_multiplier_.Resize(1,channels,1,1);
	math::Set<float,CPUContext>(channels,1.0,sum_channel_multiplier_.template mutable_data<float>(),&context_);
	sum_spatial_multiplier_.Resize(1,1,height,width);
	math::Set<float,CPUContext>(spatial_dim,1.0,sum_spatial_multiplier_.template mutable_data<float>(),&context_);
	
	if(channels_shared_){
		CAFFE_ENFORCE_EQ(scale.dim(0),1);
	}else{
		CAFFE_ENFORCE_EQ(scale.dim(0),channels);
	}

	const float* Xdata = X.data<float>();
	const float* Sdata = scale.data<float>();
	float* Ydata = Y->mutable_data<float>();
	int dim = channels*height*width;

	float* buffer_data = buffer_.mutable_data<float>();
	float* norm_data = norm->mutable_data<float>();

	math::Set<float,CPUContext>(norm->size(),float(eps_),norm_data,&context_);
	const float* sum_channel_multiplier = sum_channel_multiplier_.data<float>();
	const float* sum_spatial_multiplier = sum_spatial_multiplier_.data<float>();
//	int dim = channels*height*width;
	for(int n=0;n<num;++n){
		math::Sqr<float,CPUContext>(dim,Xdata,buffer_data,&context_);
		if(across_spatial_){
			float cpu_sum ;
			math::Sum<float,CPUContext>(dim,buffer_data,&cpu_sum,&context_);
			norm_data[n] = pow(cpu_sum+eps_,float(0.5));
			math::Scale<float,CPUContext>(dim,float(1.0/norm_data[n]),Xdata,Ydata,&context_);
		}else{
			math::Gemv<float,CPUContext>(CblasTrans,channels,spatial_dim,float(1.),
				buffer_data,sum_channel_multiplier,float(1.),
				norm_data,&context_);
			math::Powx<float,CPUContext>(spatial_dim,norm_data,float(0.5),norm_data,&context_);
			math::Gemm<float,CPUContext>(CblasNoTrans,CblasNoTrans,channels,spatial_dim,
				1,float(1.),sum_channel_multiplier,norm_data,float(0.),buffer_data,&context_);
			math::Div<float,CPUContext>(dim,Xdata,buffer_data,Ydata,&context_);
			norm_data+=spatial_dim;

		}
		if(channels_shared_){
			math::Scale<float,CPUContext>(dim,Sdata[0],Ydata,Ydata,&context_);
		}else{
			math::Gemm<float,CPUContext>(CblasNoTrans,CblasNoTrans,channels,spatial_dim,
				1,float(1.),Sdata,sum_spatial_multiplier,float(0.),buffer_data,&context_);
			math::Mul<float,CPUContext>(dim,Ydata,buffer_data,Ydata,&context_);
		}
		Xdata += dim;
		Ydata += dim;
	}
	
	return true;
}

template <>
bool NormGradientOp<float,CPUContext>::RunOnDevice()
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
	// LOG(INFO)<<"count = "<<count<<",num = "<<num<<",dim = "<<dim<<",spatial_dim = "<<spatial_dim
	// <<",channels = "<<channels<<",height = "<<height<<",width = "<<width;
	dX->ResizeLike(X);
	dscale->ResizeLike(scale);
	buffer_.Resize(1,channels,height,width);
	buffer_channel_.Resize(1,channels,1,1);
	buffer_spatial_.Resize(1,1,height,width);
	sum_channel_multiplier_.Resize(1,channels,1,1);
	sum_spatial_multiplier_.Resize(1,1,height,width);
	
	// if(across_spatial_){
	// 	norm.Resize(num,1,1,1);
	// }else{
	// 	norm.Resize(num,1,height,width);
	// }

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
	math::Set<float,CPUContext>(channels,1.0,sum_channel_multiplier,&context_);
	
	sum_spatial_multiplier_.Resize(1,1,height,width);
	
	float* sum_spatial_multiplier = sum_spatial_multiplier_.mutable_data<float>();
	math::Set<float,CPUContext>(spatial_dim,1.0,sum_spatial_multiplier,&context_);
	
	const float* norm_data = norm.data<float>();
	
	if(channels_shared_){
		float td;
		math::Dot<float,CPUContext>(count, Ydata ,dYdata, &td,&context_);
		dscaledata[0] += td / Sdata[0];
	}else{
		
		for(int n = 0; n < num; ++n){
			math::Mul<float,CPUContext>(dim, Ydata + n * dim, dYdata + n * dim , buffer_data ,&context_);
			math::Gemv<float,CPUContext>(CblasNoTrans, channels, spatial_dim, 1.0,
				buffer_data, sum_spatial_multiplier, 0.0, buffer_channel, &context_);
			math::Div<float,CPUContext>(channels, buffer_channel, Sdata, 
				buffer_channel, &context_);
			math::Add<float,CPUContext>(channels, buffer_channel, dscaledata,
				dscaledata ,&context_);
		}
		
	}
//	math::Set<float,CPUContext>(norm_.size_from_dim(0),float(2.0),norm_data,&context_);//test
	for(int n = 0; n < num; ++n){
		if(across_spatial_){
			float a;
			math::Dot<float, CPUContext>(dim, Xdata, dYdata, &a, &context_);
			math::Scale<float, CPUContext>(dim, a / norm_data[n] / norm_data[n], 
				Xdata, dXdata, &context_);
			math::Sub<float, CPUContext>(dim, dYdata, dXdata, dXdata, &context_);
			math::Scale<float, CPUContext>(dim, 1.0 / norm_data[n], dXdata, dXdata, &context_);
		}else{
			math::Mul<float, CPUContext>(dim, Xdata, dYdata, buffer_data, &context_);
			math::Gemv<float, CPUContext>(CblasTrans, channels, spatial_dim, 1.0,
				buffer_data, sum_channel_multiplier, 0.,
				buffer_spatial, &context_);
			math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 
				1, 1., sum_spatial_multiplier, buffer_spatial, 0., buffer_data, &context_);
			math::Mul<float, CPUContext>(dim, Xdata, buffer_data, dXdata, &context_);
			math::Powx<float, CPUContext>(spatial_dim, norm_data, 2.,
				buffer_spatial, &context_);
			math::Gemm<float, CPUContext>(CblasTrans, CblasNoTrans, channels, 
				spatial_dim, 1, 1.0, sum_channel_multiplier, 
				buffer_spatial, 0., buffer_data, &context_);
			math::Div<float, CPUContext>(dim, dXdata, buffer_data, dXdata, &context_);
			math::Sub<float, CPUContext>(dim, dYdata, dXdata, dXdata,&context_);
			math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, channels, 
				spatial_dim, 1, 1., sum_channel_multiplier, norm_data,
				0., buffer_data, &context_);
			math::Div<float, CPUContext>(dim, dXdata, buffer_data, dXdata, &context_);
			norm_data += spatial_dim;
		}

		if(channels_shared_){
			math::Scale<float, CPUContext>(dim, Sdata[0], dXdata, dXdata, &context_);
		}else{
			math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, channels, spatial_dim, 
				1, 1., Sdata, sum_spatial_multiplier, 0., buffer_data,&context_);
			math::Mul<float, CPUContext>(dim, dXdata, buffer_data, dXdata, &context_);
		}
		Xdata += dim;
		dYdata += dim;
		dXdata += dim;
	}
	
	//LOG(INFO)<<"norm_data = "<<norm_data[0]<<"norm_data ="<<norm_data[1];
	// float dxsum = 0.,dssum =0.;
	// for(int i = 0; i < 1478656 ; ++i){dxsum += dXdata[i];}
	// LOG(INFO)<<"dxsum = "<<dxsum;
	//// for(int i = 0; i < 1024; ++i){dssum += dscaledata[i];}
	// LOG(INFO)<<"dssum = "<<dssum;
	return true;
}

namespace{
REGISTER_CPU_OPERATOR(Norm,NormOp<float,CPUContext>);
REGISTER_CPU_OPERATOR(NormGradient,NormGradientOp<float,CPUContext>);

OPERATOR_SCHEMA(Norm)
	.NumInputs(2)
	.NumOutputs(1,2)
	.IdenticalTypeAndShape()
	.SetDoc(R"Doc()Doc")
	.Arg("across_spatial","")
	.Arg("order","")
	.Arg("eps","")
	.Arg("channel_shared","")
	.Input(0,"X","")
	.Input(1,"scale","")
	.Output(0,"Y","");

OPERATOR_SCHEMA(NormGradient).NumInputs(5).NumOutputs(2);
class GetNormGradient : public GradientMakerBase{
	using GradientMakerBase:: GradientMakerBase;
	vector<OperatorDef> GetGradientDefs() override{
		
		return SingleGradientDef(
			"NormGradient","",
			vector<string>{I(0),I(1),O(0),GO(0),O(1)},
			vector<string>{GI(0),GI(1)}
			);
	}
};
REGISTER_GRADIENT(Norm,GetNormGradient);
} // namespace 
} // namespace caffe2
