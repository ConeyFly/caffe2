import sys
import numpy as np
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')
import caffe

ssd_pt = "/home/ernie/caffe2/caffe2/python/ssd_test/detection_out/debug.prototxt"
ssd_model = "/home/ernie/ssd/caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TEST)

mbox_loc = np.load('loc_one.npy')
mbox_conf = np.load('conf_one.npy')
mbox_priorbox = np.load('prior_one.npy')

print mbox_loc.shape
print mbox_conf.shape
print mbox_priorbox.shape

ssd_net.blobs['mbox_loc'].data[...] = mbox_loc
ssd_net.blobs['mbox_conf'].data[...] = mbox_conf
ssd_net.blobs['mbox_priorbox'].data[...] = mbox_priorbox


ssd_net.forward()

detections = ssd_net.blobs['detection_out'].data

det_label = detections[0,0,:,1]
det_conf = detections[0,0,:,2]
det_xmin = detections[0,0,:,3]
det_ymin = detections[0,0,:,4]
det_xmax = detections[0,0,:,5]
det_ymax = detections[0,0,:,6]

top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.2]

print 'label = ',det_label[top_indices]
print 'conf = ',det_conf[top_indices]
print 'xmin = ',det_xmin[top_indices]
print 'ymin = ',det_ymin[top_indices]
print 'xmax = ',det_xmax[top_indices]
print 'ymax = ',det_ymax[top_indices]
