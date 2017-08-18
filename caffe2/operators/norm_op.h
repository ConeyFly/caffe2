#ifndef CAFFE2_OPERATORS_NORM_OP_H_
#define CAFFE2_OPERATORS_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"

namespace caffe2{

template <class Context>
class NormBaseOp :public Operator<Context>{
public:
	NormBaseOp(const OperatorDef& operator_def, Workspace* ws)
		:Operator<Context>(operator_def, ws){

		}
	USE_OPERATOR_CONTEXT_FUNCTIONS;
	bool RunOnDevice() override{
		return RunOnDeviceImpl();
	}
	virtual bool RunOnDeviceImpl(){
		CAFFE_NOT_IMPLEMENTED;
	}
	Tensor<Context> norm_;
};


template <typename T, class Context>
class NormOp final :public Operator<Context>{
public:
	NormOp(const OperatorDef& operator_def,Workspace* ws)
		: Operator<Context>(operator_def, ws),
		across_spatial_(OperatorBase::GetSingleArgument<bool>("across_spatial",false)),
		channels_shared_(OperatorBase::GetSingleArgument<bool>("channels_shared",false)),
		eps_(static_cast<T>(OperatorBase::template GetSingleArgument<float>("eps",1e-10))),
		order_(StringToStorageOrder(
    	OperatorBase::GetSingleArgument<string>("order","NCHW"))){
			CAFFE_ENFORCE_EQ(order_,StorageOrder::NCHW,"Only NCHW order is supported right now.");

		}
	USE_OPERATOR_CONTEXT_FUNCTIONS;
  	bool RunOnDevice() override;
protected:
//	Tensor<Context> norm_;
	Tensor<Context> sum_channel_multiplier_,sum_spatial_multiplier_;
	Tensor<Context> buffer_,buffer_channel_,buffer_spatial_;
	bool across_spatial_;
	bool channels_shared_;
	T eps_;
	StorageOrder order_;
}; 

template <typename T, class Context>
class NormGradientOp final : public Operator<Context>{
public:
	NormGradientOp(const OperatorDef& operator_def,Workspace* ws)
		: Operator<Context>(operator_def, ws),
		across_spatial_(OperatorBase::GetSingleArgument<bool>("across_spatial",false)),
		channels_shared_(OperatorBase::GetSingleArgument<bool>("channels_shared",false)),
		eps_(static_cast<T>(OperatorBase::template GetSingleArgument<float>("eps",1e-10))),
		order_(StringToStorageOrder(
    	OperatorBase::GetSingleArgument<string>("order","NCHW"))){
			CAFFE_ENFORCE_EQ(order_,StorageOrder::NCHW,"Only NCHW order is supported right now.");
		}
	USE_OPERATOR_CONTEXT_FUNCTIONS;
	bool RunOnDevice() override;

protected:
//	Tensor<Context> norm_;
	Tensor<Context> sum_channel_multiplier_,sum_spatial_multiplier_;
	Tensor<Context> buffer_,buffer_channel_,buffer_spatial_;
	bool across_spatial_;
	bool channels_shared_;
	T eps_;
	StorageOrder order_;

};


} // namespace caffe2


#endif //CAFFE2_OPERATORS_NORM_OP_H_
