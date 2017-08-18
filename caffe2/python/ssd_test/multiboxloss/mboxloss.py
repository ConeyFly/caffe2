import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

mbox_loc = np.load('mbox_loc.npy')
mbox_conf = np.load('mbox_conf.npy')
mbox_priorbox = np.load('mbox_priorbox.npy')
gt_label = np.load('gt_label.npy')

net = core.Net("mboxloss_test")
net.MultiboxLoss(["loc","conf","prior","gt_label"],["loc_pred","loc_gt","conf_pred","conf_gt"])
net.SmoothL1Loss(["loc_pred","loc_gt"],["smooth_loss"])
net.SoftmaxWithLoss(["conf_pred","conf_gt"],["P","softmax_loss"])

workspace.FeedBlob("loc",mbox_loc)
workspace.FeedBlob("conf",mbox_conf)
workspace.FeedBlob("prior",mbox_priorbox)
workspace.FeedBlob("gt_label",gt_label)

workspace.CreateNet(net.Proto())
print net.Proto()
workspace.RunNet("mboxloss_test",1)


smooth_loss = workspace.FetchBlob("smooth_loss")
softmax_loss = workspace.FetchBlob("softmax_loss")

print smooth_loss,softmax_loss
