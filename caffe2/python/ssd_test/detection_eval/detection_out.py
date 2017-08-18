import sys
import numpy as np
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')
import caffe

ssd_pt = "/home/ernie/caffe2/caffe2/python/ssd_test/detection_eval/debug.prototxt"
ssd_model = "/home/ernie/ssd/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TEST)

mbox_loc = np.load('mbox_loc.npy')
mbox_conf = np.load('mbox_conf.npy')
mbox_priorbox = np.load('mbox_priorbox.npy')
gt_label = np.load('gt_label.npy')

gt_label[0,0,0,1]=14.0
gt_label[0,0,0,5]=0.6

ssd_net.blobs['mbox_loc'].data[...] = mbox_loc
ssd_net.blobs['mbox_conf'].data[...] = mbox_conf
ssd_net.blobs['mbox_priorbox'].data[...] = mbox_priorbox
ssd_net.blobs['label'].data[...]=gt_label

ssd_net.forward()

detection_eval = ssd_net.blobs['detection'].data

np.set_printoptions(threshold=np.NaN)

detection_out = ssd_net.blobs['detection_out'].data

mbox_conf_flatten = ssd_net.blobs['mbox_conf_flatten'].data

#print mbox_conf_flatten
#print detection_out
print detection_eval
