import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

net = core.Net("detection_test")

mbox_loc = np.load('loc_one.npy')
mbox_conf = np.load('conf_one.npy')
mbox_priorbox = np.load('prior_one.npy')

print mbox_loc.shape
print mbox_conf.shape
print mbox_priorbox.shape

net.Reshape(["conf"],["conf_reshape","new_shape"],shape=[-1,21])
net.Softmax("conf_reshape","conf_softmax")
net.Reshape("conf_softmax",["conf_softmax_flat","new_shape_"],shape=[1,-1])
net.DetectionOutput(["loc","conf_softmax_flat","prior"],["detection_out"],num_classes=21,nms_threshold=0.45,keep_top_k=200,top_k_=400)


workspace.FeedBlob("loc",mbox_loc)
workspace.FeedBlob("conf",mbox_conf)
workspace.FeedBlob("prior",mbox_priorbox)


workspace.CreateNet(net.Proto())
print net.Proto()

workspace.RunNet("detection_test",1)

detections = workspace.FetchBlob('detection_out')

print detections.shape

det_label = detections[0,0,:,1]
det_conf = detections[0,0,:,2]
det_xmin = detections[0,0,:,3]
det_ymin = detections[0,0,:,4]
det_xmax = detections[0,0,:,5]
det_ymax = detections[0,0,:,6]

top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]

print 'label = ',det_label[top_indices]
print 'conf = ',det_conf[top_indices]
print 'xmin = ',det_xmin[top_indices]
print 'ymin = ',det_ymin[top_indices]
print 'xmax = ',det_xmax[top_indices]
print 'ymax = ',det_ymax[top_indices]
