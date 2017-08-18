#ifndef CAFFE2_OPERATORS_TEST_DETECTIONS_OP_H
#define CAFFE2_OPERATORS_TEST_DETECTIONS_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

#include <map>
using namespace std;

namespace caffe2{

template<typename T, class Context>
class TestDetectionOp final : public Operator<Context>{
public:
	TestDetectionOp(const OperatorDef& operator_def, Workspace* ws)
		:Operator<Context>(operator_def, ws),
		iter_size_(OperatorBase::GetSingleArgument<int>("iter_size",1)),
		test_size_(OperatorBase::GetSingleArgument<int>("test_size",4952)),
		ap_version_(OperatorBase::GetSingleArgument<string>("ap_version","11_point"))
		{
			count_ = 0;
		}
		USE_OPERATOR_CONTEXT_FUNCTIONS;
		bool RunOnDevice() override;
protected:
	int count_;
	int iter_size_;
	int test_size_;
	string ap_version_;
	map<int, map<int, vector<pair<float, int> > > > all_true_pos;
  	map<int, map<int, vector<pair<float, int> > > > all_false_pos;
  	map<int, map<int, int> > all_num_pos;
};

}


#endif