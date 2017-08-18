#include "caffe2/operators/prior_box_op.h"


namespace caffe2{


namespace{
REGISTER_CPU_OPERATOR(PriorBox,PriorBoxOp<float,CPUContext>);
NO_GRADIENT(PriorBox);

OPERATOR_SCHEMA(PriorBox)
	.NumInputs(2)
	.NumOutputs(1)
	.TensorInferenceFunction(
		[](const OperatorDef& def,const vector<TensorShape>& in){
			ArgumentHelper helper(def);
			const StorageOrder order = StringToStorageOrder(
				helper.GetSingleArgument<string>("order","NCHW"));
			const TensorShape &X = in[0];
			
			int layer_height = X.dims(2);
			int layer_width = X.dims(3);

			vector<float> aspect_ratios = helper.GetRepeatedArgument<float>("aspect_ratios");
			vector<float> min_sizes = helper.GetRepeatedArgument<float>("min_sizes");
			vector<float> new_ar;
			new_ar.push_back(1.);
			bool flip = helper.GetSingleArgument<bool>("flip",true);

			for(int i = 0;i <aspect_ratios.size(); ++i ){
				float ar = aspect_ratios[i];
				bool already_exist = false;
				for(int j = 0;j < new_ar.size(); ++j ){
					if(fabs(ar-new_ar[j]) < 1e-6){
						already_exist = true;
						break;
					}
				}
				if(!already_exist){
					new_ar.push_back(ar);
					if(flip){
						new_ar.push_back(1./ar);
					}
				}
			}
			int num_priors = new_ar.size()*min_sizes.size();
			vector<float> max_sizes = helper.GetRepeatedArgument<float>("max_sizes");
			for(int i=0;i<max_sizes.size();i++)
				num_priors += 1;

			TensorShape Y = CreateTensorShape(
				vector<int>({1,2,layer_width*layer_height*num_priors*4}),X.data_type());
			return vector<TensorShape>({Y});
		})
	.SetDoc(R"Doc()Doc")
	.Arg("min_sizes","")
	.Arg("max_sizes","")
	.Arg("aspect_ratios","")
	.Arg("flip","flip or not")
	.Arg("clip","clip box")
	.Arg("variance","variance")
	.Arg("img_size","img size")
	.Arg("img_w","img width")
	.Arg("img_h","img height")
	.Arg("step","step ")
	.Arg("step_w","step width")
	.Arg("step_h","step height")
	.Arg("offset","offset")
	.Arg("order","NCHW")
	.Input(0,"X","NCHW tensor")
	.Input(1,"data","NCHW tensor")
	.Output(0,"Y","prior boxes");
}

}
