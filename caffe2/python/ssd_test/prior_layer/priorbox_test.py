import sys
sys.path.insert(0,'/home/ernie/caffe2/build')
from caffe2.python import cnn, workspace, core
from caffe2.proto import caffe2_pb2
import numpy as np
import time

net = core.Net("prior_test")
net.PriorBox(["conv4_3_norm", "data"], "prior_box", min_sizes=[21.0,30.0], max_sizes=[45.0,60.0],
             aspect_ratios=[2.0,3.0], flip=True, clip=False,
             variance=[0.1, 0.1, 0.2, 0.2], step=8.)
print net.Proto()

workspace.FeedBlob("data", np.random.rand(1, 3, 300, 300).astype(np.float32))
workspace.FeedBlob("conv4_3_norm", np.random.rand(1, 512, 38, 38).astype(np.float32))

workspace.CreateNet(net.Proto())
workspace.RunNet("prior_test", 1)

caffe_out = np.load('./prior.npy')
caffe2_out = workspace.FetchBlob("prior_box")
#import ipdb; ipdb.set_trace()
print np.allclose(caffe_out, caffe2_out)

print caffe_out
print caffe2_out

print np.sum(caffe2_out-caffe_out)/70000
