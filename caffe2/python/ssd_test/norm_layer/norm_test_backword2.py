import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

workspace.ResetWorkspace()

device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
net = core.Net("norm_backward1")
net.NormGradient(["X","scale","Y","dY","norm"],["dX","dscale"],across_spatial=False,
	channels_shared=False,eps=1e-10,device_option=device_opts)

X = np.load('X.npy')
scale = np.load('scale.npy')
Y = np.load('Y.npy')
dY = np.load('dY.npy')

norm = np.ones((2, 1, 38, 38),dtype=np.float32)
#scale = np.ones((512),dtype=np.float32)


workspace.FeedBlob("X",X,device_option=device_opts)
workspace.FeedBlob("scale",scale,device_option=device_opts)
workspace.FeedBlob("Y",Y,device_option=device_opts)
workspace.FeedBlob("dY",dY,device_option=device_opts)
workspace.FeedBlob("norm",norm,device_option=device_opts)

workspace.CreateNet(net.Proto())
workspace.RunNet("norm_backward1",1)

dX = workspace.FetchBlob('dX')
dscale = workspace.FetchBlob('dscale')

#print dX
print dscale
#print scale.shape

#print scale
#print workspace.CurrentWorkspace()

#print workspace.Blobs()
