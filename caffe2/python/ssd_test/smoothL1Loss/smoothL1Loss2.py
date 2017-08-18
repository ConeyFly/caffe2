import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

net = core.Net("smoothL1Loss_test")
net.SmoothL1Loss(["data1","data2"],"loss")

print net.Proto()

data1 = np.load('data1.npy')
data2 = np.load('data2.npy')
caffe_out = np.load('loss_data.npy')

workspace.FeedBlob("data1",data1)
workspace.FeedBlob("data2",data2)

workspace.CreateNet(net.Proto())

workspace.RunNet("smoothL1Loss_test",1)

caffe2_out = workspace.FetchBlob('loss')

print (caffe_out)

print '****'

print (caffe2_out)

#print data1-data2
#print data2


#print np.allclose(conv4_3_norm_out1,conv4_3_norm_out2)

