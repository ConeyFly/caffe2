import sys
import numpy as np
sys.path.insert(0,'/home/ernie/testssd/caffe-ssd/python')
import caffe
ssd_pt = '/home/ernie/caffe2/caffe2/python/ssd_test/norm_layer/debug.prototxt'
ssd_model = '/home/ernie/testssd/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

ssd_net = caffe.Net(ssd_pt,ssd_model,caffe.TRAIN)

# data_blob = np.random.rand(1, 512, 38, 38).astype(np.float32)
# ssd_net.blobs['conv4_3'].data[...] = data_blob
# ssd_scale = ssd_net.params['conv4_3_norm'][0].data
# ssd_net.forward()
# norm_data = ssd_net.blobs['conv4_3_norm'].data
# np.save('./data.npy', data_blob)
# np.save('./scale.npy', ssd_scale)
# np.save('./conv4_3_norm_out.npy', norm_data)

top_diff = np.random.rand(1, 512, 38, 38).astype(np.float32) * 1
top_data = np.random.rand(1, 512, 38, 38).astype(np.float32) * 1
bottom_data = np.random.rand(1, 512, 38, 38).astype(np.float32) * 1
bottom_diff = np.random.rand(1, 512, 38, 38).astype(np.float32) * 1
#scale_diff = np.random.rand(512).astype(np.float32)* 10000
scale_data = np.random.rand(512).astype(np.float32)* 1
#norm_data = np.load('conv4_3_norm_out.npy'
ssd_net.blobs['conv4_3_norm'].data[...] = top_data
ssd_net.blobs['conv4_3_norm'].diff[...] = top_diff
ssd_net.blobs['conv4_3'].data[...] = bottom_data
ssd_net.blobs['conv4_3'].diff[...] = bottom_diff
#scale_data = ssd_net.params['conv4_3_norm'][0].data
#print scale_data
#scale_diff = ssd_net.params['conv4_3_norm'][0].diff
#print scale_diff
#print np.shape(scale_diff),np.shape(scale_data)

ssd_net.params['conv4_3_norm'][0].data[...] = scale_data
#ssd_net.params['conv4_3_norm'][0].diff[...] = scale_diff

#ssd_net.forward()

#bottom_diff = ssd_net.blobs['conv4_3'].d

print np.sum(ssd_net.blobs['conv4_3_norm'].diff - top_diff)
print np.sum(ssd_net.blobs['conv4_3_norm'].data - top_data)
print np.sum(ssd_net.blobs['conv4_3'].diff - bottom_diff)
print np.sum(ssd_net.blobs['conv4_3'].data - bottom_data)
print np.sum(ssd_net.params['conv4_3_norm'][0].diff[...])
#print np.sum(ssd_net.params['conv4_3_norm'][0].diff[...] - scale_diff)
print np.sum(ssd_net.params['conv4_3_norm'][0].data[...] )


print '#####'
ssd_net.backward()

print np.sum(ssd_net.blobs['conv4_3_norm'].diff - top_diff)
print np.sum(ssd_net.blobs['conv4_3_norm'].data - top_data)
print np.sum(ssd_net.blobs['conv4_3'].diff - bottom_diff)
print np.sum(ssd_net.blobs['conv4_3'].data - bottom_data)

print np.sum(ssd_net.params['conv4_3_norm'][0].diff[...] )
print np.sum(ssd_net.params['conv4_3_norm'][0].data[...])
print np.sum(ssd_net.blobs['conv4_3'].diff)

np.save('dY.npy',top_diff)
np.save('Y.npy',top_data)
np.save('X.npy',bottom_data)
np.save('dX.npy',bottom_diff)
np.save('scale.npy',scale_data)

np.save('dY_new.npy',ssd_net.blobs['conv4_3_norm'].diff)
np.save('Y_new.npy',ssd_net.blobs['conv4_3_norm'].data)
np.save('dX_new.npy',ssd_net.blobs['conv4_3'].diff)
np.save('X_new.npy',ssd_net.blobs['conv4_3'].data)
np.save('scale_diff.npy',ssd_net.params['conv4_3_norm'][0].diff)


#print bottom_diff
#print ssd_net.blobs['conv4_3'].data
