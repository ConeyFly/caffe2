#include "caffe2/operators/prior_box_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2{

namespace{
REGISTER_CUDA_OPERATOR(PriorBox,PriorBoxOp<float,CUDAContext>);

}

}
