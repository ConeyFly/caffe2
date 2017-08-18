import sys
import numpy as np
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')
import caffe
path = '/home/ernie/testssd/caffe-ssd/'

ssd_pt = '/home/ernie/caffe2/caffe2/python/ssd_test/prior_layer/debug.prototxt'
ssd_model = '/home/ernie/testssd/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TEST)

conv4_norm_blob = np.random.rand(1,512,38,38).astype(np.float32)
data_blob = np.random.rand(1,3,300,300).astype(np.float32)

ssd_net.blobs['data'].data[...] = data_blob
ssd_net.blobs['conv4_norm'].data[...] = conv4_norm_blob

ssd_net.forward()

prior_data = ssd_net.blobs['conv4_3_norm_mbox_priorbox'].data

np.save('prior.npy',prior_data)
