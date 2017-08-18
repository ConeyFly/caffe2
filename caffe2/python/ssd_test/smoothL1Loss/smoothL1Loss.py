import sys
import numpy as np
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')
import caffe
ssd_pt = '/home/ernie/caffe2/caffe2/python/ssd_test/smoothL1Loss/debug.prototxt'
ssd_model = '/home/ernie/testssd/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TEST)

data1 = np.random.rand(1, 512, 38, 38).astype(np.float32)
data2 = np.random.rand(1, 512, 38, 38).astype(np.float32)
ssd_net.blobs['data1'].data[...] = data1
ssd_net.blobs['data2'].data[...] = data2
ssd_net.forward()
loss_data = ssd_net.blobs['loss'].data
np.save('./data1.npy', data1)
np.save('./data2.npy', data2)
np.save('./loss_data.npy', loss_data)



