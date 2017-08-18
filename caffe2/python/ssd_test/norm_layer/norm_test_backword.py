import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

workspace.ResetWorkspace()
net = core.Net("norm_backward")
net.NormGradient(["X","scale","Y","dY","norm"],["dX","dscale"],across_spatial=False,
	channels_shared=False,eps=1e-10)

X = np.load('X.npy')
scale = np.load('scale.npy')
Y = np.load('Y.npy')
dY = np.load('dY.npy')
norm = np.ones((2, 1, 38, 38),dtype=np.float32)

workspace.FeedBlob("X",X)
workspace.FeedBlob("scale",scale)
workspace.FeedBlob("Y",Y)
workspace.FeedBlob("dY",dY)
workspace.FeedBlob("norm",norm)

workspace.CreateNet(net.Proto())
workspace.RunNet("norm_backward",1)

dX = workspace.FetchBlob('dX')
dscale = workspace.FetchBlob('dscale')



print dX
print dscale
print scale.shape
