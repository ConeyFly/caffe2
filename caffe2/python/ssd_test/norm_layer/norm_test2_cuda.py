import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
net = core.Net("norm_test")
net.Norm(["conv4_3","scale"],["norm","n"],across_spatial=False,channels_shared=False,eps=1e-10,device_option=device_opts)

print net.Proto()

data = np.load('data.npy')
scale = np.load('scale.npy')
conv4_3_norm_out1 = np.load('conv4_3_norm_out.npy')

workspace.FeedBlob("conv4_3",data,device_option=device_opts)
workspace.FeedBlob("scale",scale,device_option=device_opts)

workspace.CreateNet(net.Proto())

workspace.RunNet("norm_test",1)

conv4_3_norm_out2 = workspace.FetchBlob('norm')

print (conv4_3_norm_out1)

print '****'

print (conv4_3_norm_out2)



#print np.allclose(conv4_3_norm_out1,conv4_3_norm_out2)

