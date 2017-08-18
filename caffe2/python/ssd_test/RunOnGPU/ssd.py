from matplotlib import pyplot
import sys
sys.path.insert(0,'/home/ernie/caffe2/build')

from caffe2.python import core, model_helper, net_drawer, workspace, brew
from caffe2.proto import caffe2_pb2
import caffe2.python.models.SSDNet as SSDNet
import numpy as np


def AddInput(model, batch_size, db, db_type):
    data, gt_label = model.net.AnnotationInput(
        [],["Annodata","gt_label"],batch_size = batch_size,
        db=db, db_type = db_type,scale = 300, crop = 300,
        use_gpu_transform = 0, mirror = 1, warp = 1, color = 1,
        use_caffe_datum = 1,
        interp_mode = ["LINEAR","AREA","NEAREST","CUBIC","LANCZOS4"],
        decode_threads=4
    )
    data_gpu = model.net.CopyCPUToGPU(data,'data_gpu')
    data_gpu = model.StopGradient(data_gpu, data_gpu)
    return data_gpu, gt_label

def AddSSDNetModel(model,data):
    mbox_layers = SSDNet.SSDNet(model,data)
    return mbox_layers

def AddConfReshape(model, conf, num_classes):
    model.net.Reshape([conf],["conf_reshape","shape_1"],shape=[-1,num_classes])
    model.net.Softmax("conf_reshape","conf_softmax")
    conf_softmax_flat=model.net.Reshape("conf_softmax",["conf_softmax_flat","shape_2"],shape=[1,-1])
    return conf_softmax_flat


def AddTrainingOperators(model, mbox_layers, gt_label):
    MultiboxInput = mbox_layers
   # MultiboxInput.append(gt_label)
    
    mbox_loc_cpu = model.net.CopyGPUToCPU(mbox_layers[0],'mbox_loc_cpu')
    mbox_conf_cpu = model.net.CopyGPUToCPU(mbox_layers[1],'mbox_conf_cpu')
   
    loc_pred,loc_gt,conf_pred,conf_gt=model.net.MultiboxLoss(
    [mbox_loc_cpu, mbox_conf_cpu, mbox_layers[2],gt_label],['loc_pred','loc_gt','conf_pred','conf_gt']
    )
   # loc_pred,loc_gt,conf_pred,conf_gt=model.net.MultiboxLoss(
   # [mbox_layers[0], mbox_layers[1], mbox_layers[2],gt_label],['loc_pred','loc_gt','conf_pred','conf_gt']
   # )
    loc_pred_gpu = model.net.CopyCPUToGPU(loc_pred,'loc_pred_gpu')
    loc_gt_gpu = model.net.CopyCPUToGPU(loc_gt,'loc_gt_gpu')
    conf_pred_gpu = model.net.CopyCPUToGPU(conf_pred,'conf_pred_gpu')
    conf_gt_gpu = model.net.CopyCPUToGPU(conf_gt,'conf_gt_gpu')
    SmoothL1Loss = model.net.SmoothL1Loss([loc_pred_gpu, loc_gt_gpu],'SmoothL1Loss')
    P,SoftmaxWithLoss = model.net.SoftmaxWithLoss([conf_pred_gpu,conf_gt_gpu],["P","SoftmaxWithLoss"])
    model.AddGradientOperators([SmoothL1Loss,SoftmaxWithLoss])
    
    ITER = brew.iter(model,"iter")
  #  ITER = model.param_init_net.ConstantFill([],'ITER',shape=[1],value=0,dtype=core.DataType.INT32)
    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.001,policy="step",stepsize=80000,gamma=0.1
        )
    ONE = model.param_init_net.ConstantFill([],"ONE",shape=[1],value=1.0)
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.WeightedSum([param, ONE, param_grad, LR],param)



train_model = model_helper.ModelHelper(name="ssd_train")
data, gt_label = AddInput(train_model, batch_size=2, 
                         db='/home/ernie/ssd/caffe/examples/VOC0712/VOC0712_trainval_lmdb',
                         db_type='lmdb')
mbox_layers = AddSSDNetModel(train_model, data)
AddTrainingOperators(train_model, mbox_layers, gt_label)
#train_model.RunAllOnGPU()
deploy_model = model_helper.ModelHelper(name="ssd_deploy", init_params=False)
mbox_layers = AddSSDNetModel(deploy_model, "data")
conf_softmax_flat = AddConfReshape(deploy_model, mbox_layers[1], 21)
mbox_layers[1] = 'conf_softmax_flat'
deploy_model.net.DetectionOutput(mbox_layers,["detection_out"]
                                ,num_classes=21,nms_threshold=0.45,keep_top_k=200,top_k_=400)
net_def = caffe2_pb2.NetDef()
with open('/home/ernie/caffemodel/VGG16_init_net.pb','r') as f:
    net_def.ParseFromString(f.read())
workspace.RunNetOnce(net_def)

def deviceOpts(cpu_or_cuda):
    device_opts = caffe2_pb2.DeviceOption()
    if cpu_or_cuda == 'cpu':
        device_opts.device_type = caffe2_pb2.CPU
    elif cpu_or_cuda == 'cuda':
        device_opts.device_type = caffe2_pb2.CUDA
        device_opts.cuda_gpu_id = 0
    return device_opts

for op in train_model.net.Proto().op:
    if op.type =='AnnotationInput':
        op.device_option.CopyFrom(deviceOpts('cpu'))
    elif op.type == 'CopyCPUToGPU':
        op.device_option.CopyFrom(deviceOpts('cuda'))
    elif op.type == 'Concat' and op.output[0] == 'mbox_priorbox':
        op.device_option.CopyFrom(deviceOpts('cpu'))
    elif op.type == 'MultiboxLoss':
        op.device_option.CopyFrom(deviceOpts('cpu'))
    elif op.type == 'MultiboxLossGradient':
        op.device_option.CopyFrom(deviceOpts('cpu'))
    else:
        op.device_option.CopyFrom(deviceOpts('cuda'))



for op in train_model.param_init_net.Proto().op:
    if op.output[0] == "iter":
        op.device_option.CopyFrom(deviceOpts('cpu'))
    elif op.output[0] == "ONE":
        op.device_option.CopyFrom(deviceOpts('cuda'))
    else: op.device_option.CopyFrom(deviceOpts('cuda'))
 #   if op.output[0] =='ITER':
  #      op.device_option.CopyFrom(deviceOpts('cpu'))


losses_ = []
def UpdateSmoothedLoss(loss, iter_, average_loss, smoothed_loss):
    if len(losses_) < average_loss:
        losses_.append(loss)
        size = len(losses_)
        smoothed_loss = (smoothed_loss * (size - 1) + loss) / size
    else:
        idx = iter_ % average_loss
        smoothed_loss = smoothed_loss + (loss - losses_[idx]) / average_loss
        losses_[idx] = loss
    return smoothed_loss
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite=True)
total_iters = 50000
loc_loss = np.zeros(total_iters)
conf_loss = np.zeros(total_iters)
smoothed_loss = 0
for i in range(total_iters):
    workspace.RunNet(train_model.net)
    loc_loss[i] = workspace.FetchBlob('SmoothL1Loss')
    conf_loss[i] = workspace.FetchBlob('SoftmaxWithLoss')
    smoothed_loss = UpdateSmoothedLoss(loc_loss[i]+conf_loss[i], i ,10,smoothed_loss)
    if i%500==0:
        print 'iters = ',i,'loss = ',smoothed_loss,'smooth = ',loc_loss[i],'softmax = ',conf_loss[i]

