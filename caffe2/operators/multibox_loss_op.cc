#include "caffe2/operators/multibox_loss_op.h"

namespace caffe2{


namespace{

REGISTER_CPU_OPERATOR(MultiboxLoss,MultiboxLossOp<float,CPUContext>);
REGISTER_CPU_OPERATOR(MultiboxLossGradient,MultiboxLossGradientOp<float,CPUContext>);

OPERATOR_SCHEMA(MultiboxLoss)
	.NumInputs(4)
	.NumOutputs(4)
	.SetDoc(R"DOC()DOC")
	.Arg("num_classes","")
	.Arg("share_location","")
	.Arg("scale","")
	.Input(0,"loc","")
	.Input(1,"conf","")
	.Input(2,"prior","")
	.Input(3,"gt","")
	.Output(0,"loc_pred","")
	.Output(1,"loc_gt","")
	.Output(2,"conf_pred","")
	.Output(3,"conf_gt","");

OPERATOR_SCHEMA(MultiboxLossGradient).NumInputs(6).NumOutputs(2);
class GetMultiboxLossGradient : public GradientMakerBase{
    using GradientMakerBase::GradientMakerBase;
    vector<OperatorDef> GetGradientDefs() override{
	//CHECK_EQ(def_.input_size(),6);
	return SingleGradientDef(
	   "MultiboxLossGradient","",
	   vector<string>{I(0),I(1),I(2),I(3),GO(0),GO(2)},
	   vector<string>{GI(0),GI(1)}
	);
    }
};

REGISTER_GRADIENT(MultiboxLoss,GetMultiboxLossGradient);
}// namespace

} // namespace caffe2
