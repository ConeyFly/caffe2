import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

net = core.Net("detection_eval_test")

mbox_loc = np.load('mbox_loc.npy')
mbox_conf = np.load('mbox_conf.npy')
mbox_priorbox = np.load('mbox_priorbox.npy')
gt_label = np.load('gt_label.npy')

print mbox_loc.shape
print mbox_conf.shape
print mbox_priorbox.shape
print gt_label

gt_label[0,0,0,1]=14.0
gt_label[0,0,0,5]=0.6
print gt_label

net.Reshape(["conf"],["conf_reshape","new_shape"],shape=[-1,21])
net.Softmax("conf_reshape","conf_softmax")
net.Reshape("conf_softmax",["conf_softmax_flat","new_shape_"],shape=[1,-1])
net.DetectionOutput(["loc","conf_softmax_flat","prior"],['detection_out'],num_classes=21,nms_threshold=0.45,keep_top_k=200,top_k=400)

#detection_out = np.load('detections.npy')
#print detection_out.shape
net.DetectionEvalute(['detection_out','gt_label'],['detection_eval'],num_classes=21,overlap_threshold=0.001,resize_valid=False,
name_size_file='/home/ernie/caffe2/caffe2/python/ssd_test/detection_eval/test_name_size.txt')


workspace.FeedBlob("loc",mbox_loc)
workspace.FeedBlob("conf",mbox_conf)
workspace.FeedBlob("prior",mbox_priorbox)
workspace.FeedBlob("gt_label",gt_label)
#workspace.FeedBlob('detection_out',detection_out)

workspace.CreateNet(net.Proto())
print net.Proto()

workspace.RunNet("detection_eval_test",1)

conf_softmax_flat = workspace.FetchBlob('conf_softmax_flat')
detections = workspace.FetchBlob('detection_out')
detection_eval = workspace.FetchBlob('detection_eval')

np.set_printoptions(threshold=np.NaN)

#print conf_softmax_flat
#print detections
print detection_eval


