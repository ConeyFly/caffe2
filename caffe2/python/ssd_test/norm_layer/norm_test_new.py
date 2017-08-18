import sys
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')

import caffe
import numpy as np

ssd_pt = '/home/ernie/caffe2/caffe2/python/ssd_test/norm_layer/debug.prototxt'
ssd_model = '/home/ernie/testssd/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TRAIN)
# gen data

#X = np.random.rand(2, 512, 38, 38).astype(np.float32)
#scale = np.random.rand(512).astype(np.float32)
data = np.load('data.npy')
scale = np.load('scale.npy')
#dY = np.random.rand(2, 512, 38, 38).astype(np.float32)

ssd_net.blobs['conv4_3'].data[...] = data
ssd_net.params['conv4_3_norm'][0].data[...] = scale

ssd_net.forward()


Y = ssd_net.blobs['conv4_3_norm'].data 

print Y



