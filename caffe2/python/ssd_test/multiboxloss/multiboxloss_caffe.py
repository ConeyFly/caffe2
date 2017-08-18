import sys
import numpy as np
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')
import caffe

ssd_pt = "/home/ernie/caffe2/caffe2/python/ssd_test/multiboxloss/debug.prototxt"
ssd_model = "/home/ernie/testssd/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel"

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TEST)

mbox_loc = np.load('mbox_loc.npy')
mbox_conf = np.load('mbox_conf.npy')
mbox_priorbox = np.load('mbox_priorbox.npy')
gt_label = np.load('gt_label.npy')


ssd_net.blobs['mbox_loc'].data[...] = mbox_loc
ssd_net.blobs['mbox_conf'].data[...] = mbox_conf
ssd_net.blobs['mbox_priorbox'].data[...] = mbox_priorbox
ssd_net.blobs['gt_label'].data[...] = gt_label

ssd_net.forward()
mbox_loss = ssd_net.blobs['mbox_loss']
#print ssd_net.blobs['mbox_loss'].data

ssd_net.backward()
dloc = ssd_net.blobs['mbox_loc'].diff
dconf = ssd_net.blobs['mbox_conf'].diff

np.set_printoptions(threshold=np.NaN)

print np.shape(dloc)
print (dconf[dconf!=0.])

#np.save('dloc.npy',dloc)
#np.save('dconf.npy',dconf)

