#ifndef CAFFE2_OPERATORS_SMOOTH_L1_LOSS_OP_H_
#define CAFFE2_OPERATORS_SMOOTH_L1_LOSS_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2{

template <typename T,class Context>
class SmoothL1LossOp final : public Operator<Context>{
public:
	SmoothL1LossOp(const OperatorDef& operator_def,Workspace* ws)
		:Operator<Context>(operator_def,ws),
		has_weights_(OperatorBase::GetSingleArgument<bool>("has_weights",0)),
		order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order","NCHW"))){
			CAFFE_ENFORCE_EQ(order_,StorageOrder::NCHW,"Only NCHW order is supported right now.");
		}
		
	USE_OPERATOR_CONTEXT_FUNCTIONS;

	bool RunOnDevice() override;
protected:
	Tensor<Context> diff_;
	Tensor<Context> errors_;
	bool has_weights_;
	StorageOrder order_;
};

template <typename T,class Context>
class SmoothL1LossGradientOp final : public Operator<Context>{
public:
	SmoothL1LossGradientOp(const OperatorDef& operator_def,Workspace* ws)
		:Operator<Context>(operator_def,ws),
		has_weights_(OperatorBase::GetSingleArgument<bool>("has_weights",0)),
		order_(StringToStorageOrder(OperatorBase::GetSingleArgument<string>("order","NCHW"))){
			CAFFE_ENFORCE_EQ(order_,StorageOrder::NCHW,"Only NCHW order is supported right now.");
		}
		
	USE_OPERATOR_CONTEXT_FUNCTIONS;

	bool RunOnDevice() override;
protected:
	Tensor<Context> diff_;
	Tensor<Context> errors_;
	bool has_weights_;
	StorageOrder order_;
};

}; //namespace

#endif
