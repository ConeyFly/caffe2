import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time


#device_opts = caffe2_pb2.DeviceOption()
#device_opts.device_type = caffe2_pb2.CUDA
#device_opts.cuda_gpu_id = 0
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
net = core.Net("smoothL1Loss_test")
net.SmoothL1Loss(["data1","data2"],"loss",device_option=device_opts)

print net.Proto()

data1 = np.load('data1.npy')
data2 = np.load('data2.npy')

print np.shape(data1)

caffe_out = np.load('loss_data.npy')

workspace.FeedBlob("data1",data1,device_option=device_opts)
workspace.FeedBlob("data2",data2,device_option=device_opts)



workspace.CreateNet(net.Proto())



workspace.RunNet("smoothL1Loss_test",1)

print "ok3"

caffe2_out = workspace.FetchBlob('loss')

print (caffe_out)

print '****'

print (caffe2_out)

#print data1-data2
#print data2


#print np.allclose(conv4_3_norm_out1,conv4_3_norm_out2)

