name: "ssd_train"
op {
  output: "Annodata"
  output: "gt_label"
  name: ""
  type: "AnnotationInput"
  arg {
    name: "scale"
    i: 300
  }
  arg {
    name: "use_caffe_datum"
    i: 1
  }
  arg {
    name: "color"
    i: 1
  }
  arg {
    name: "use_gpu_transform"
    i: 0
  }
  arg {
    name: "db"
    s: "/home/ernie/data/VOCdevkit/test/lmdb/test_trainval_lmdb"
  }
  arg {
    name: "crop"
    i: 300
  }
  arg {
    name: "interp_mode"
    strings: "LINEAR"
    strings: "AREA"
    strings: "NEAREST"
    strings: "CUBIC"
    strings: "LANCZOS4"
  }
  arg {
    name: "warp"
    i: 1
  }
  arg {
    name: "batch_size"
    i: 1
  }
  arg {
    name: "mirror"
    i: 1
  }
  arg {
    name: "db_type"
    s: "lmdb"
  }
}
op {
  input: "Annodata"
  output: "data"
  name: ""
  type: "StopGradient"
}
op {
  input: "gt_label"
  output: "gt_label"
  name: ""
  type: "StopGradient"
}
op {
  input: "data"
  input: "conv1_1_w"
  input: "conv1_1_b"
  output: "conv1_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 64
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 3
  }
}
op {
  input: "conv1_1"
  output: "conv1_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv1_1"
  input: "conv1_2_w"
  input: "conv1_2_b"
  output: "conv1_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 64
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 64
  }
}
op {
  input: "conv1_2"
  output: "conv1_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv1_2"
  output: "pool1"
  name: ""
  type: "MaxPool"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
}
op {
  input: "pool1"
  input: "conv2_1_w"
  input: "conv2_1_b"
  output: "conv2_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 64
  }
}
op {
  input: "conv2_1"
  output: "conv2_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv2_1"
  input: "conv2_2_w"
  input: "conv2_2_b"
  output: "conv2_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 128
  }
}
op {
  input: "conv2_2"
  output: "conv2_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv2_2"
  output: "pool2"
  name: ""
  type: "MaxPool"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
}
op {
  input: "pool2"
  input: "conv3_1_w"
  input: "conv3_1_b"
  output: "conv3_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 128
  }
}
op {
  input: "conv3_1"
  output: "conv3_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv3_1"
  input: "conv3_2_w"
  input: "conv3_2_b"
  output: "conv3_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv3_2"
  output: "conv3_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv3_2"
  input: "conv3_3_w"
  input: "conv3_3_b"
  output: "conv3_3"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv3_3"
  output: "conv3_3"
  name: ""
  type: "Relu"
}
op {
  input: "conv3_3"
  output: "pool3"
  name: ""
  type: "MaxPool"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
}
op {
  input: "pool3"
  input: "conv4_1_w"
  input: "conv4_1_b"
  output: "conv4_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv4_1"
  output: "conv4_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv4_1"
  input: "conv4_2_w"
  input: "conv4_2_b"
  output: "conv4_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv4_2"
  output: "conv4_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv4_2"
  input: "conv4_3_w"
  input: "conv4_3_b"
  output: "conv4_3"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv4_3"
  output: "conv4_3"
  name: ""
  type: "Relu"
}
op {
  input: "conv4_3"
  output: "pool4"
  name: ""
  type: "MaxPool"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
}
op {
  input: "pool4"
  input: "conv5_1_w"
  input: "conv5_1_b"
  output: "conv5_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv5_1"
  output: "conv5_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv5_1"
  input: "conv5_2_w"
  input: "conv5_2_b"
  output: "conv5_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv5_2"
  output: "conv5_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv5_2"
  input: "conv5_3_w"
  input: "conv5_3_b"
  output: "conv5_3"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv5_3"
  output: "conv5_3"
  name: ""
  type: "Relu"
}
op {
  input: "conv5_3"
  output: "pool5"
  name: ""
  type: "MaxPool"
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "stride"
    i: 1
  }
}
op {
  input: "pool5"
  input: "fc6_w"
  input: "fc6_b"
  output: "fc6"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 1024
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "fc6"
  output: "fc6"
  name: ""
  type: "Relu"
}
op {
  input: "fc6"
  input: "fc7_w"
  input: "fc7_b"
  output: "fc7"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 1024
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 1024
  }
}
op {
  input: "fc7"
  output: "fc7"
  name: ""
  type: "Relu"
}
op {
  input: "fc7"
  input: "conv6_1_w"
  input: "conv6_1_b"
  output: "conv6_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 1024
  }
}
op {
  input: "conv6_1"
  output: "conv6_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv6_1"
  input: "conv6_2_w"
  input: "conv6_2_b"
  output: "conv6_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  arg {
    name: "stride"
    i: 2
  }
}
op {
  input: "conv6_2"
  output: "conv6_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv6_2"
  input: "conv7_1_w"
  input: "conv7_1_b"
  output: "conv7_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 512
  }
  arg {
    name: "stride"
    i: 1
  }
}
op {
  input: "conv7_1"
  output: "conv7_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv7_1"
  input: "conv7_2_w"
  input: "conv7_2_b"
  output: "conv7_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 128
  }
  arg {
    name: "stride"
    i: 2
  }
}
op {
  input: "conv7_2"
  output: "conv7_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv7_2"
  input: "conv8_1_w"
  input: "conv8_1_b"
  output: "conv8_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 256
  }
  arg {
    name: "stride"
    i: 1
  }
}
op {
  input: "conv8_1"
  output: "conv8_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv8_1"
  input: "conv8_2_w"
  input: "conv8_2_b"
  output: "conv8_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 128
  }
  arg {
    name: "stride"
    i: 1
  }
}
op {
  input: "conv8_2"
  output: "conv8_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv8_2"
  input: "conv9_1_w"
  input: "conv9_1_b"
  output: "conv9_1"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 256
  }
  arg {
    name: "stride"
    i: 1
  }
}
op {
  input: "conv9_1"
  output: "conv9_1"
  name: ""
  type: "Relu"
}
op {
  input: "conv9_1"
  input: "conv9_2_w"
  input: "conv9_2_b"
  output: "conv9_2"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 128
  }
  arg {
    name: "stride"
    i: 1
  }
}
op {
  input: "conv9_2"
  output: "conv9_2"
  name: ""
  type: "Relu"
}
op {
  input: "conv4_3"
  input: "scale"
  output: "conv4_3_norm"
  output: "scale_new"
  name: ""
  type: "Norm"
  arg {
    name: "across_spatial"
    i: 0
  }
  arg {
    name: "channels_shared"
    i: 0
  }
}
op {
  input: "conv4_3_norm"
  input: "conv4_3_norm_mbox_loc_w"
  input: "conv4_3_norm_mbox_loc_b"
  output: "conv4_3_norm_mbox_loc"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 16
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv4_3_norm_mbox_loc"
  output: "conv4_3_norm_mbox_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv4_3_norm_mbox_perm"
  output: "conv4_3_norm_mbox_loc_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv4_3_norm"
  input: "conv4_3_norm_mbox_conf_w"
  input: "conv4_3_norm_mbox_conf_b"
  output: "conv4_3_norm_conf"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 84
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv4_3_norm_conf"
  output: "conv4_3_norm_mbox_conf_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv4_3_norm_mbox_conf_perm"
  output: "conv4_3_norm_mbox_conf_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv4_3_norm"
  input: "data"
  output: "conv4_3_norm_mbox_priorbox"
  name: ""
  type: "PriorBox"
  arg {
    name: "aspect_ratios"
    floats: 2.0
  }
  arg {
    name: "clip"
    i: 0
  }
  arg {
    name: "min_sizes"
    floats: 30.0
  }
  arg {
    name: "flip"
    i: 1
  }
  arg {
    name: "step"
    f: 8.0
  }
  arg {
    name: "offset"
    f: 0.5
  }
  arg {
    name: "max_sizes"
    floats: 60.0
  }
  arg {
    name: "variance"
    floats: 0.1
    floats: 0.1
    floats: 0.2
    floats: 0.2
  }
}
op {
  input: "conv4_3_norm_mbox_priorbox"
  output: "conv4_3_norm_mbox_priorbox"
  name: ""
  type: "StopGradient"
}
op {
  input: "fc7"
  input: "fc7_mbox_loc_w"
  input: "fc7_mbox_loc_b"
  output: "fc7_mbox_loc"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 24
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 1024
  }
}
op {
  input: "fc7_mbox_loc"
  output: "fc7_mbox_loc_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "fc7_mbox_loc_perm"
  output: "fc7_mbox_loc_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "fc7"
  input: "fc7_mbox_conf_w"
  input: "fc7_mbox_conf_b"
  output: "fc7_mbox_conf"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 126
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 1024
  }
}
op {
  input: "fc7_mbox_conf"
  output: "fc7_mbox_conf_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "fc7_mbox_conf_perm"
  output: "fc7_mbox_conf_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "fc7"
  input: "data"
  output: "fc7_mbox_priorbox"
  name: ""
  type: "PriorBox"
  arg {
    name: "aspect_ratios"
    floats: 2.0
    floats: 3.0
  }
  arg {
    name: "clip"
    i: 0
  }
  arg {
    name: "min_sizes"
    floats: 60.0
  }
  arg {
    name: "flip"
    i: 1
  }
  arg {
    name: "step"
    f: 16.0
  }
  arg {
    name: "offset"
    f: 0.5
  }
  arg {
    name: "max_sizes"
    floats: 111.0
  }
  arg {
    name: "variance"
    floats: 0.1
    floats: 0.1
    floats: 0.2
    floats: 0.2
  }
}
op {
  input: "fc7_mbox_priorbox"
  output: "fc7_mbox_priorbox"
  name: ""
  type: "StopGradient"
}
op {
  input: "conv6_2"
  input: "conv6_2_mbox_loc_w"
  input: "conv6_2_mbox_loc_b"
  output: "conv6_2_mbox_loc"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 24
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv6_2_mbox_loc"
  output: "conv6_2_mbox_loc_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv6_2_mbox_loc_perm"
  output: "conv6_2_mbox_loc_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv6_2"
  input: "conv6_2_mbox_conf_w"
  input: "conv6_2_mbox_conf_b"
  output: "conv6_2_mbox_conf"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 126
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
}
op {
  input: "conv6_2_mbox_conf"
  output: "conv6_2_mbox_conf_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv6_2_mbox_conf_perm"
  output: "conv6_2_mbox_conf_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv6_2"
  input: "data"
  output: "conv6_2_mbox_priorbox"
  name: ""
  type: "PriorBox"
  arg {
    name: "aspect_ratios"
    floats: 2.0
    floats: 3.0
  }
  arg {
    name: "clip"
    i: 0
  }
  arg {
    name: "min_sizes"
    floats: 111.0
  }
  arg {
    name: "flip"
    i: 1
  }
  arg {
    name: "step"
    f: 32.0
  }
  arg {
    name: "offset"
    f: 0.5
  }
  arg {
    name: "max_sizes"
    floats: 162.0
  }
  arg {
    name: "variance"
    floats: 0.1
    floats: 0.1
    floats: 0.2
    floats: 0.2
  }
}
op {
  input: "conv6_2_mbox_priorbox"
  output: "conv6_2_mbox_priorbox"
  name: ""
  type: "StopGradient"
}
op {
  input: "conv7_2"
  input: "conv7_2_mbox_loc_w"
  input: "conv7_2_mbox_loc_b"
  output: "conv7_2_mbox_loc"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 24
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv7_2_mbox_loc"
  output: "conv7_2_mbox_loc_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv7_2_mbox_loc_perm"
  output: "conv7_2_mbox_loc_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv7_2"
  input: "conv7_2_mbox_conf_w"
  input: "conv7_2_mbox_conf_b"
  output: "conv7_2_mbox_conf"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 126
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv7_2_mbox_conf"
  output: "conv7_2_mbox_conf_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv7_2_mbox_conf_perm"
  output: "conv7_2_mbox_conf_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv7_2"
  input: "data"
  output: "conv7_2_mbox_priorbox"
  name: ""
  type: "PriorBox"
  arg {
    name: "aspect_ratios"
    floats: 2.0
    floats: 3.0
  }
  arg {
    name: "clip"
    i: 0
  }
  arg {
    name: "min_sizes"
    floats: 162.0
  }
  arg {
    name: "flip"
    i: 1
  }
  arg {
    name: "step"
    f: 64.0
  }
  arg {
    name: "offset"
    f: 0.5
  }
  arg {
    name: "max_sizes"
    floats: 213.0
  }
  arg {
    name: "variance"
    floats: 0.1
    floats: 0.1
    floats: 0.2
    floats: 0.2
  }
}
op {
  input: "conv7_2_mbox_priorbox"
  output: "conv7_2_mbox_priorbox"
  name: ""
  type: "StopGradient"
}
op {
  input: "conv8_2"
  input: "conv8_2_mbox_loc_w"
  input: "conv8_2_mbox_loc_b"
  output: "conv8_2_mbox_loc"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 16
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv8_2_mbox_loc"
  output: "conv8_2_mbox_loc_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv8_2_mbox_loc_perm"
  output: "conv8_2_mbox_loc_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv8_2"
  input: "conv8_2_mbox_conf_w"
  input: "conv8_2_mbox_conf_b"
  output: "conv8_2_mbox_conf"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 84
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv8_2_mbox_conf"
  output: "conv8_2_mbox_conf_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv8_2_mbox_conf_perm"
  output: "conv8_2_mbox_conf_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv8_2"
  input: "data"
  output: "conv8_2_mbox_priorbox"
  name: ""
  type: "PriorBox"
  arg {
    name: "aspect_ratios"
    floats: 2.0
  }
  arg {
    name: "clip"
    i: 0
  }
  arg {
    name: "min_sizes"
    floats: 213.0
  }
  arg {
    name: "flip"
    i: 1
  }
  arg {
    name: "step"
    f: 100.0
  }
  arg {
    name: "offset"
    f: 0.5
  }
  arg {
    name: "max_sizes"
    floats: 264.0
  }
  arg {
    name: "variance"
    floats: 0.1
    floats: 0.1
    floats: 0.2
    floats: 0.2
  }
}
op {
  input: "conv8_2_mbox_priorbox"
  output: "conv8_2_mbox_priorbox"
  name: ""
  type: "StopGradient"
}
op {
  input: "conv9_2"
  input: "conv9_2_mbox_loc_w"
  input: "conv9_2_mbox_loc_b"
  output: "conv9_2_mbox_loc"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 16
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv9_2_mbox_loc"
  output: "conv9_2_mbox_loc_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv9_2_mbox_loc_perm"
  output: "conv9_2_mbox_loc_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv9_2"
  input: "conv9_2_mbox_conf_w"
  input: "conv9_2_mbox_conf_b"
  output: "conv9_2_mbox_conf"
  name: ""
  type: "Conv"
  arg {
    name: "dim_out"
    i: 84
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
}
op {
  input: "conv9_2_mbox_conf"
  output: "conv9_2_mbox_conf_perm"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 2
    ints: 3
    ints: 1
  }
}
op {
  input: "conv9_2_mbox_conf_perm"
  output: "conv9_2_mbox_conf_flat"
  name: ""
  type: "Flatten"
}
op {
  input: "conv9_2"
  input: "data"
  output: "conv9_2_mbox_priorbox"
  name: ""
  type: "PriorBox"
  arg {
    name: "aspect_ratios"
    floats: 2.0
  }
  arg {
    name: "clip"
    i: 0
  }
  arg {
    name: "min_sizes"
    floats: 264.0
  }
  arg {
    name: "flip"
    i: 1
  }
  arg {
    name: "step"
    f: 300.0
  }
  arg {
    name: "offset"
    f: 0.5
  }
  arg {
    name: "max_sizes"
    floats: 315.0
  }
  arg {
    name: "variance"
    floats: 0.1
    floats: 0.1
    floats: 0.2
    floats: 0.2
  }
}
op {
  input: "conv9_2_mbox_priorbox"
  output: "conv9_2_mbox_priorbox"
  name: ""
  type: "StopGradient"
}
op {
  input: "conv4_3_norm_mbox_loc_flat"
  input: "fc7_mbox_loc_flat"
  input: "conv6_2_mbox_loc_flat"
  input: "conv7_2_mbox_loc_flat"
  input: "conv8_2_mbox_loc_flat"
  input: "conv9_2_mbox_loc_flat"
  output: "mbox_loc"
  output: "mbox_loc_axes"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "conv4_3_norm_mbox_conf_flat"
  input: "fc7_mbox_conf_flat"
  input: "conv6_2_mbox_conf_flat"
  input: "conv7_2_mbox_conf_flat"
  input: "conv8_2_mbox_conf_flat"
  input: "conv9_2_mbox_conf_flat"
  output: "mbox_conf"
  output: "mbox_conf_axes"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 1
  }
}
op {
  input: "conv4_3_norm_mbox_priorbox"
  input: "fc7_mbox_priorbox"
  input: "conv6_2_mbox_priorbox"
  input: "conv7_2_mbox_priorbox"
  input: "conv8_2_mbox_priorbox"
  input: "conv9_2_mbox_priorbox"
  output: "mbox_priorbox"
  output: "mbox_priorbox_axes"
  name: ""
  type: "Concat"
  arg {
    name: "axis"
    i: 2
  }
}
op {
  input: "mbox_loc"
  input: "mbox_conf"
  input: "mbox_priorbox"
  input: "gt_label"
  output: "loc_pred"
  output: "loc_gt"
  output: "conf_pred"
  output: "conf_gt"
  name: ""
  type: "MultiboxLoss"
}
op {
  input: "loc_pred"
  input: "loc_gt"
  output: "SmoothL1Loss"
  name: ""
  type: "SmoothL1Loss"
}
op {
  input: "conf_pred"
  input: "conf_gt"
  output: "P"
  output: "SoftmaxWithLoss"
  name: ""
  type: "SoftmaxWithLoss"
}
op {
  input: "SoftmaxWithLoss"
  output: "SoftmaxWithLoss_autogen_grad"
  name: ""
  type: "ConstantFill"
  arg {
    name: "value"
    f: 1.0
  }
}
op {
  input: "SmoothL1Loss"
  output: "SmoothL1Loss_autogen_grad"
  name: ""
  type: "ConstantFill"
  arg {
    name: "value"
    f: 1.0
  }
}
op {
  input: "conf_pred"
  input: "conf_gt"
  input: "P"
  input: "SoftmaxWithLoss_autogen_grad"
  output: "conf_pred_grad"
  name: ""
  type: "SoftmaxWithLossGradient"
  is_gradient_op: true
}
op {
  input: "loc_pred"
  input: "loc_gt"
  input: "SmoothL1Loss_autogen_grad"
  output: "loc_pred_grad"
  name: ""
  type: "SmoothL1LossGradient"
  is_gradient_op: true
}
op {
  input: "mbox_loc"
  input: "mbox_conf"
  input: "mbox_priorbox"
  input: "gt_label"
  input: "loc_pred_grad"
  input: "conf_pred_grad"
  output: "mbox_loc_grad"
  output: "mbox_conf_grad"
  name: ""
  type: "MultiboxLossGradient"
  is_gradient_op: true
}
op {
  input: "mbox_conf_grad"
  input: "mbox_conf_axes"
  output: "conv4_3_norm_mbox_conf_flat_grad"
  output: "fc7_mbox_conf_flat_grad"
  output: "conv6_2_mbox_conf_flat_grad"
  output: "conv7_2_mbox_conf_flat_grad"
  output: "conv8_2_mbox_conf_flat_grad"
  output: "conv9_2_mbox_conf_flat_grad"
  name: ""
  type: "Split"
  arg {
    name: "axis"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "mbox_loc_grad"
  input: "mbox_loc_axes"
  output: "conv4_3_norm_mbox_loc_flat_grad"
  output: "fc7_mbox_loc_flat_grad"
  output: "conv6_2_mbox_loc_flat_grad"
  output: "conv7_2_mbox_loc_flat_grad"
  output: "conv8_2_mbox_loc_flat_grad"
  output: "conv9_2_mbox_loc_flat_grad"
  name: ""
  type: "Split"
  arg {
    name: "axis"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv9_2_mbox_conf_flat_grad"
  input: "conv9_2_mbox_conf_perm"
  output: "conv9_2_mbox_conf_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv9_2_mbox_conf_perm_grad"
  output: "conv9_2_mbox_conf_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv9_2"
  input: "conv9_2_mbox_conf_w"
  input: "conv9_2_mbox_conf_grad"
  output: "conv9_2_mbox_conf_w_grad"
  output: "conv9_2_mbox_conf_b_grad"
  output: "conv9_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 84
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv9_2_mbox_loc_flat_grad"
  input: "conv9_2_mbox_loc_perm"
  output: "conv9_2_mbox_loc_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv9_2_mbox_loc_perm_grad"
  output: "conv9_2_mbox_loc_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv9_2"
  input: "conv9_2_mbox_loc_w"
  input: "conv9_2_mbox_loc_grad"
  output: "conv9_2_mbox_loc_w_grad"
  output: "conv9_2_mbox_loc_b_grad"
  output: "_conv9_2_grad_autosplit_0"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 16
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv9_2_grad"
  input: "_conv9_2_grad_autosplit_0"
  output: "conv9_2_grad"
  name: ""
  type: "Sum"
}
op {
  input: "conv8_2_mbox_conf_flat_grad"
  input: "conv8_2_mbox_conf_perm"
  output: "conv8_2_mbox_conf_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv8_2_mbox_conf_perm_grad"
  output: "conv8_2_mbox_conf_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv8_2"
  input: "conv8_2_mbox_conf_w"
  input: "conv8_2_mbox_conf_grad"
  output: "conv8_2_mbox_conf_w_grad"
  output: "conv8_2_mbox_conf_b_grad"
  output: "conv8_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 84
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv8_2_mbox_loc_flat_grad"
  input: "conv8_2_mbox_loc_perm"
  output: "conv8_2_mbox_loc_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv8_2_mbox_loc_perm_grad"
  output: "conv8_2_mbox_loc_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv8_2"
  input: "conv8_2_mbox_loc_w"
  input: "conv8_2_mbox_loc_grad"
  output: "conv8_2_mbox_loc_w_grad"
  output: "conv8_2_mbox_loc_b_grad"
  output: "_conv8_2_grad_autosplit_0"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 16
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv7_2_mbox_conf_flat_grad"
  input: "conv7_2_mbox_conf_perm"
  output: "conv7_2_mbox_conf_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv7_2_mbox_conf_perm_grad"
  output: "conv7_2_mbox_conf_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv7_2"
  input: "conv7_2_mbox_conf_w"
  input: "conv7_2_mbox_conf_grad"
  output: "conv7_2_mbox_conf_w_grad"
  output: "conv7_2_mbox_conf_b_grad"
  output: "conv7_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 126
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv7_2_mbox_loc_flat_grad"
  input: "conv7_2_mbox_loc_perm"
  output: "conv7_2_mbox_loc_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv7_2_mbox_loc_perm_grad"
  output: "conv7_2_mbox_loc_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv7_2"
  input: "conv7_2_mbox_loc_w"
  input: "conv7_2_mbox_loc_grad"
  output: "conv7_2_mbox_loc_w_grad"
  output: "conv7_2_mbox_loc_b_grad"
  output: "_conv7_2_grad_autosplit_0"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 24
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv6_2_mbox_conf_flat_grad"
  input: "conv6_2_mbox_conf_perm"
  output: "conv6_2_mbox_conf_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv6_2_mbox_conf_perm_grad"
  output: "conv6_2_mbox_conf_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv6_2"
  input: "conv6_2_mbox_conf_w"
  input: "conv6_2_mbox_conf_grad"
  output: "conv6_2_mbox_conf_w_grad"
  output: "conv6_2_mbox_conf_b_grad"
  output: "conv6_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 126
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv6_2_mbox_loc_flat_grad"
  input: "conv6_2_mbox_loc_perm"
  output: "conv6_2_mbox_loc_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv6_2_mbox_loc_perm_grad"
  output: "conv6_2_mbox_loc_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv6_2"
  input: "conv6_2_mbox_loc_w"
  input: "conv6_2_mbox_loc_grad"
  output: "conv6_2_mbox_loc_w_grad"
  output: "conv6_2_mbox_loc_b_grad"
  output: "_conv6_2_grad_autosplit_0"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 24
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "fc7_mbox_conf_flat_grad"
  input: "fc7_mbox_conf_perm"
  output: "fc7_mbox_conf_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "fc7_mbox_conf_perm_grad"
  output: "fc7_mbox_conf_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "fc7"
  input: "fc7_mbox_conf_w"
  input: "fc7_mbox_conf_grad"
  output: "fc7_mbox_conf_w_grad"
  output: "fc7_mbox_conf_b_grad"
  output: "fc7_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 126
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 1024
  }
  is_gradient_op: true
}
op {
  input: "fc7_mbox_loc_flat_grad"
  input: "fc7_mbox_loc_perm"
  output: "fc7_mbox_loc_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "fc7_mbox_loc_perm_grad"
  output: "fc7_mbox_loc_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "fc7"
  input: "fc7_mbox_loc_w"
  input: "fc7_mbox_loc_grad"
  output: "fc7_mbox_loc_w_grad"
  output: "fc7_mbox_loc_b_grad"
  output: "_fc7_grad_autosplit_0"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 24
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 1024
  }
  is_gradient_op: true
}
op {
  input: "conv4_3_norm_mbox_conf_flat_grad"
  input: "conv4_3_norm_mbox_conf_perm"
  output: "conv4_3_norm_mbox_conf_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv4_3_norm_mbox_conf_perm_grad"
  output: "conv4_3_norm_conf_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv4_3_norm"
  input: "conv4_3_norm_mbox_conf_w"
  input: "conv4_3_norm_conf_grad"
  output: "conv4_3_norm_mbox_conf_w_grad"
  output: "conv4_3_norm_mbox_conf_b_grad"
  output: "conv4_3_norm_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 84
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv4_3_norm_mbox_loc_flat_grad"
  input: "conv4_3_norm_mbox_perm"
  output: "conv4_3_norm_mbox_perm_grad"
  name: ""
  type: "ResizeLike"
  is_gradient_op: true
}
op {
  input: "conv4_3_norm_mbox_perm_grad"
  output: "conv4_3_norm_mbox_loc_grad"
  name: ""
  type: "Transpose"
  arg {
    name: "axes"
    ints: 0
    ints: 3
    ints: 1
    ints: 2
  }
  is_gradient_op: true
}
op {
  input: "conv4_3_norm"
  input: "conv4_3_norm_mbox_loc_w"
  input: "conv4_3_norm_mbox_loc_grad"
  output: "conv4_3_norm_mbox_loc_w_grad"
  output: "conv4_3_norm_mbox_loc_b_grad"
  output: "_conv4_3_norm_grad_autosplit_0"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 16
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv4_3_norm_grad"
  input: "_conv4_3_norm_grad_autosplit_0"
  output: "conv4_3_norm_grad"
  name: ""
  type: "Sum"
}
op {
  input: "conv4_3"
  input: "scale"
  input: "conv4_3_norm"
  input: "conv4_3_norm_grad"
  input: "scale_new"
  output: "conv4_3_grad"
  output: "scale_grad"
  name: ""
  type: "NormGradient"
  arg {
    name: "across_spatial"
    i: 0
  }
  arg {
    name: "channels_shared"
    i: 0
  }
  is_gradient_op: true
}
op {
  input: "conv9_2"
  input: "conv9_2_grad"
  output: "conv9_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv9_1"
  input: "conv9_2_w"
  input: "conv9_2_grad"
  output: "conv9_2_w_grad"
  output: "conv9_2_b_grad"
  output: "conv9_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 128
  }
  arg {
    name: "stride"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv9_1"
  input: "conv9_1_grad"
  output: "conv9_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv8_2"
  input: "conv9_1_w"
  input: "conv9_1_grad"
  output: "conv9_1_w_grad"
  output: "conv9_1_b_grad"
  output: "_conv8_2_grad_autosplit_1"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 256
  }
  arg {
    name: "stride"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv8_2_grad"
  input: "_conv8_2_grad_autosplit_0"
  input: "_conv8_2_grad_autosplit_1"
  output: "conv8_2_grad"
  name: ""
  type: "Sum"
}
op {
  input: "conv8_2"
  input: "conv8_2_grad"
  output: "conv8_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv8_1"
  input: "conv8_2_w"
  input: "conv8_2_grad"
  output: "conv8_2_w_grad"
  output: "conv8_2_b_grad"
  output: "conv8_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 128
  }
  arg {
    name: "stride"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv8_1"
  input: "conv8_1_grad"
  output: "conv8_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv7_2"
  input: "conv8_1_w"
  input: "conv8_1_grad"
  output: "conv8_1_w_grad"
  output: "conv8_1_b_grad"
  output: "_conv7_2_grad_autosplit_1"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 256
  }
  arg {
    name: "stride"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv7_2_grad"
  input: "_conv7_2_grad_autosplit_0"
  input: "_conv7_2_grad_autosplit_1"
  output: "conv7_2_grad"
  name: ""
  type: "Sum"
}
op {
  input: "conv7_2"
  input: "conv7_2_grad"
  output: "conv7_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv7_1"
  input: "conv7_2_w"
  input: "conv7_2_grad"
  output: "conv7_2_w_grad"
  output: "conv7_2_b_grad"
  output: "conv7_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 128
  }
  arg {
    name: "stride"
    i: 2
  }
  is_gradient_op: true
}
op {
  input: "conv7_1"
  input: "conv7_1_grad"
  output: "conv7_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv6_2"
  input: "conv7_1_w"
  input: "conv7_1_grad"
  output: "conv7_1_w_grad"
  output: "conv7_1_b_grad"
  output: "_conv6_2_grad_autosplit_1"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 512
  }
  arg {
    name: "stride"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv6_2_grad"
  input: "_conv6_2_grad_autosplit_0"
  input: "_conv6_2_grad_autosplit_1"
  output: "conv6_2_grad"
  name: ""
  type: "Sum"
}
op {
  input: "conv6_2"
  input: "conv6_2_grad"
  output: "conv6_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv6_1"
  input: "conv6_2_w"
  input: "conv6_2_grad"
  output: "conv6_2_w_grad"
  output: "conv6_2_b_grad"
  output: "conv6_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  arg {
    name: "stride"
    i: 2
  }
  is_gradient_op: true
}
op {
  input: "conv6_1"
  input: "conv6_1_grad"
  output: "conv6_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "fc7"
  input: "conv6_1_w"
  input: "conv6_1_grad"
  output: "conv6_1_w_grad"
  output: "conv6_1_b_grad"
  output: "_fc7_grad_autosplit_1"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "pad"
    i: 0
  }
  arg {
    name: "dim_in"
    i: 1024
  }
  is_gradient_op: true
}
op {
  input: "fc7_grad"
  input: "_fc7_grad_autosplit_0"
  input: "_fc7_grad_autosplit_1"
  output: "fc7_grad"
  name: ""
  type: "Sum"
}
op {
  input: "fc7"
  input: "fc7_grad"
  output: "fc7_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "fc6"
  input: "fc7_w"
  input: "fc7_grad"
  output: "fc7_w_grad"
  output: "fc7_b_grad"
  output: "fc6_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 1024
  }
  arg {
    name: "kernel"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 1024
  }
  is_gradient_op: true
}
op {
  input: "fc6"
  input: "fc6_grad"
  output: "fc6_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "pool5"
  input: "fc6_w"
  input: "fc6_grad"
  output: "fc6_w_grad"
  output: "fc6_b_grad"
  output: "pool5_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 1024
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv5_3"
  input: "pool5"
  input: "pool5_grad"
  output: "conv5_3_grad"
  name: ""
  type: "MaxPoolGradient"
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "stride"
    i: 1
  }
  is_gradient_op: true
}
op {
  input: "conv5_3"
  input: "conv5_3_grad"
  output: "conv5_3_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv5_2"
  input: "conv5_3_w"
  input: "conv5_3_grad"
  output: "conv5_3_w_grad"
  output: "conv5_3_b_grad"
  output: "conv5_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv5_2"
  input: "conv5_2_grad"
  output: "conv5_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv5_1"
  input: "conv5_2_w"
  input: "conv5_2_grad"
  output: "conv5_2_w_grad"
  output: "conv5_2_b_grad"
  output: "conv5_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv5_1"
  input: "conv5_1_grad"
  output: "conv5_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "pool4"
  input: "conv5_1_w"
  input: "conv5_1_grad"
  output: "conv5_1_w_grad"
  output: "conv5_1_b_grad"
  output: "pool4_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv4_3"
  input: "pool4"
  input: "pool4_grad"
  output: "_conv4_3_grad_autosplit_0"
  name: ""
  type: "MaxPoolGradient"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
  is_gradient_op: true
}
op {
  input: "conv4_3_grad"
  input: "_conv4_3_grad_autosplit_0"
  output: "conv4_3_grad"
  name: ""
  type: "Sum"
}
op {
  input: "conv4_3"
  input: "conv4_3_grad"
  output: "conv4_3_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv4_2"
  input: "conv4_3_w"
  input: "conv4_3_grad"
  output: "conv4_3_w_grad"
  output: "conv4_3_b_grad"
  output: "conv4_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv4_2"
  input: "conv4_2_grad"
  output: "conv4_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv4_1"
  input: "conv4_2_w"
  input: "conv4_2_grad"
  output: "conv4_2_w_grad"
  output: "conv4_2_b_grad"
  output: "conv4_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 512
  }
  is_gradient_op: true
}
op {
  input: "conv4_1"
  input: "conv4_1_grad"
  output: "conv4_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "pool3"
  input: "conv4_1_w"
  input: "conv4_1_grad"
  output: "conv4_1_w_grad"
  output: "conv4_1_b_grad"
  output: "pool3_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 512
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv3_3"
  input: "pool3"
  input: "pool3_grad"
  output: "conv3_3_grad"
  name: ""
  type: "MaxPoolGradient"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
  is_gradient_op: true
}
op {
  input: "conv3_3"
  input: "conv3_3_grad"
  output: "conv3_3_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv3_2"
  input: "conv3_3_w"
  input: "conv3_3_grad"
  output: "conv3_3_w_grad"
  output: "conv3_3_b_grad"
  output: "conv3_2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv3_2"
  input: "conv3_2_grad"
  output: "conv3_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv3_1"
  input: "conv3_2_w"
  input: "conv3_2_grad"
  output: "conv3_2_w_grad"
  output: "conv3_2_b_grad"
  output: "conv3_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 256
  }
  is_gradient_op: true
}
op {
  input: "conv3_1"
  input: "conv3_1_grad"
  output: "conv3_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "pool2"
  input: "conv3_1_w"
  input: "conv3_1_grad"
  output: "conv3_1_w_grad"
  output: "conv3_1_b_grad"
  output: "pool2_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 256
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 128
  }
  is_gradient_op: true
}
op {
  input: "conv2_2"
  input: "pool2"
  input: "pool2_grad"
  output: "conv2_2_grad"
  name: ""
  type: "MaxPoolGradient"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
  is_gradient_op: true
}
op {
  input: "conv2_2"
  input: "conv2_2_grad"
  output: "conv2_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv2_1"
  input: "conv2_2_w"
  input: "conv2_2_grad"
  output: "conv2_2_w_grad"
  output: "conv2_2_b_grad"
  output: "conv2_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 128
  }
  is_gradient_op: true
}
op {
  input: "conv2_1"
  input: "conv2_1_grad"
  output: "conv2_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "pool1"
  input: "conv2_1_w"
  input: "conv2_1_grad"
  output: "conv2_1_w_grad"
  output: "conv2_1_b_grad"
  output: "pool1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 128
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 64
  }
  is_gradient_op: true
}
op {
  input: "conv1_2"
  input: "pool1"
  input: "pool1_grad"
  output: "conv1_2_grad"
  name: ""
  type: "MaxPoolGradient"
  arg {
    name: "kernel"
    i: 2
  }
  arg {
    name: "stride"
    i: 2
  }
  is_gradient_op: true
}
op {
  input: "conv1_2"
  input: "conv1_2_grad"
  output: "conv1_2_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "conv1_1"
  input: "conv1_2_w"
  input: "conv1_2_grad"
  output: "conv1_2_w_grad"
  output: "conv1_2_b_grad"
  output: "conv1_1_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 64
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 64
  }
  is_gradient_op: true
}
op {
  input: "conv1_1"
  input: "conv1_1_grad"
  output: "conv1_1_grad"
  name: ""
  type: "ReluGradient"
  is_gradient_op: true
}
op {
  input: "data"
  input: "conv1_1_w"
  input: "conv1_1_grad"
  output: "conv1_1_w_grad"
  output: "conv1_1_b_grad"
  output: "data_grad"
  name: ""
  type: "ConvGradient"
  arg {
    name: "dim_out"
    i: 64
  }
  arg {
    name: "kernel"
    i: 3
  }
  arg {
    name: "pad"
    i: 1
  }
  arg {
    name: "dim_in"
    i: 3
  }
  is_gradient_op: true
}
op {
  input: "ITER"
  output: "ITER"
  name: ""
  type: "Iter"
}
op {
  input: "ITER"
  output: "LR"
  name: ""
  type: "LearningRate"
  arg {
    name: "policy"
    s: "step"
  }
  arg {
    name: "stepsize"
    i: 80000
  }
  arg {
    name: "base_lr"
    f: -0.001
  }
  arg {
    name: "gamma"
    f: 0.1
  }
}
op {
  input: "conv8_2_mbox_loc_flat"
  input: "ONE"
  input: "conv8_2_mbox_loc_flat_grad"
  input: "LR"
  output: "conv8_2_mbox_loc_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_w"
  input: "ONE"
  input: "conv9_2_w_grad"
  input: "LR"
  output: "conv9_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_loc_w"
  input: "ONE"
  input: "fc7_mbox_loc_w_grad"
  input: "LR"
  output: "fc7_mbox_loc_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_loc_b"
  input: "ONE"
  input: "conv8_2_mbox_loc_b_grad"
  input: "LR"
  output: "conv8_2_mbox_loc_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_conf"
  input: "ONE"
  input: "conv8_2_mbox_conf_grad"
  input: "LR"
  output: "conv8_2_mbox_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv1_1"
  input: "ONE"
  input: "conv1_1_grad"
  input: "LR"
  output: "conv1_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_loc_b"
  input: "ONE"
  input: "fc7_mbox_loc_b_grad"
  input: "LR"
  output: "fc7_mbox_loc_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_loc_w"
  input: "ONE"
  input: "conv8_2_mbox_loc_w_grad"
  input: "LR"
  output: "conv8_2_mbox_loc_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_conf"
  input: "ONE"
  input: "conv7_2_mbox_conf_grad"
  input: "LR"
  output: "conv7_2_mbox_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_loc_b"
  input: "ONE"
  input: "conv6_2_mbox_loc_b_grad"
  input: "LR"
  output: "conv6_2_mbox_loc_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_conf_flat"
  input: "ONE"
  input: "conv8_2_mbox_conf_flat_grad"
  input: "LR"
  output: "conv8_2_mbox_conf_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "mbox_conf"
  input: "ONE"
  input: "mbox_conf_grad"
  input: "LR"
  output: "mbox_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_conf_perm"
  input: "ONE"
  input: "conv7_2_mbox_conf_perm_grad"
  input: "LR"
  output: "conv7_2_mbox_conf_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_conf_perm"
  input: "ONE"
  input: "conv8_2_mbox_conf_perm_grad"
  input: "LR"
  output: "conv8_2_mbox_conf_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_b"
  input: "ONE"
  input: "conv6_2_b_grad"
  input: "LR"
  output: "conv6_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_loc_w"
  input: "ONE"
  input: "conv6_2_mbox_loc_w_grad"
  input: "LR"
  output: "conv6_2_mbox_loc_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_w"
  input: "ONE"
  input: "conv6_2_w_grad"
  input: "LR"
  output: "conv6_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "pool2"
  input: "ONE"
  input: "pool2_grad"
  input: "LR"
  output: "pool2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "pool1"
  input: "ONE"
  input: "pool1_grad"
  input: "LR"
  output: "pool1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_conf_flat"
  input: "ONE"
  input: "conv7_2_mbox_conf_flat_grad"
  input: "LR"
  output: "conv7_2_mbox_conf_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "pool4"
  input: "ONE"
  input: "pool4_grad"
  input: "LR"
  output: "pool4"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_loc_w"
  input: "ONE"
  input: "conv4_3_norm_mbox_loc_w_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_loc_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_conf_perm"
  input: "ONE"
  input: "conv4_3_norm_mbox_conf_perm_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_conf_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_w"
  input: "ONE"
  input: "fc7_w_grad"
  input: "LR"
  output: "fc7_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_perm"
  input: "ONE"
  input: "conv4_3_norm_mbox_perm_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_loc_b"
  input: "ONE"
  input: "conv4_3_norm_mbox_loc_b_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_loc_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_1_w"
  input: "ONE"
  input: "conv3_1_w_grad"
  input: "LR"
  output: "conv3_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_b"
  input: "ONE"
  input: "fc7_b_grad"
  input: "LR"
  output: "fc7_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_conf_flat"
  input: "ONE"
  input: "fc7_mbox_conf_flat_grad"
  input: "LR"
  output: "fc7_mbox_conf_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv2_2_b"
  input: "ONE"
  input: "conv2_2_b_grad"
  input: "LR"
  output: "conv2_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_conf_b"
  input: "ONE"
  input: "conv9_2_mbox_conf_b_grad"
  input: "LR"
  output: "conv9_2_mbox_conf_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_3_w"
  input: "ONE"
  input: "conv3_3_w_grad"
  input: "LR"
  output: "conv3_3_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_loc"
  input: "ONE"
  input: "conv7_2_mbox_loc_grad"
  input: "LR"
  output: "conv7_2_mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv2_2_w"
  input: "ONE"
  input: "conv2_2_w_grad"
  input: "LR"
  output: "conv2_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_conf"
  input: "ONE"
  input: "conv9_2_mbox_conf_grad"
  input: "LR"
  output: "conv9_2_mbox_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_loc"
  input: "ONE"
  input: "conv8_2_mbox_loc_grad"
  input: "LR"
  output: "conv8_2_mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_3_b"
  input: "ONE"
  input: "conv3_3_b_grad"
  input: "LR"
  output: "conv3_3_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_1_b"
  input: "ONE"
  input: "conv3_1_b_grad"
  input: "LR"
  output: "conv3_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_conf_w"
  input: "ONE"
  input: "conv9_2_mbox_conf_w_grad"
  input: "LR"
  output: "conv9_2_mbox_conf_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_loc_flat"
  input: "ONE"
  input: "conv6_2_mbox_loc_flat_grad"
  input: "LR"
  output: "conv6_2_mbox_loc_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_conf"
  input: "ONE"
  input: "conv4_3_norm_conf_grad"
  input: "LR"
  output: "conv4_3_norm_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_1_b"
  input: "ONE"
  input: "conv6_1_b_grad"
  input: "LR"
  output: "conv6_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_1_b"
  input: "ONE"
  input: "conv7_1_b_grad"
  input: "LR"
  output: "conv7_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_loc_perm"
  input: "ONE"
  input: "conv6_2_mbox_loc_perm_grad"
  input: "LR"
  output: "conv6_2_mbox_loc_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_loc_b"
  input: "ONE"
  input: "conv9_2_mbox_loc_b_grad"
  input: "LR"
  output: "conv9_2_mbox_loc_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_conf_w"
  input: "ONE"
  input: "fc7_mbox_conf_w_grad"
  input: "LR"
  output: "fc7_mbox_conf_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_1_w"
  input: "ONE"
  input: "conv7_1_w_grad"
  input: "LR"
  output: "conv7_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_1_w"
  input: "ONE"
  input: "conv6_1_w_grad"
  input: "LR"
  output: "conv6_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv1_2"
  input: "ONE"
  input: "conv1_2_grad"
  input: "LR"
  output: "conv1_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_conf_w"
  input: "ONE"
  input: "conv4_3_norm_mbox_conf_w_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_conf_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_loc_flat"
  input: "ONE"
  input: "conv7_2_mbox_loc_flat_grad"
  input: "LR"
  output: "conv7_2_mbox_loc_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_conf_w"
  input: "ONE"
  input: "conv6_2_mbox_conf_w_grad"
  input: "LR"
  output: "conv6_2_mbox_conf_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_loc"
  input: "ONE"
  input: "conv6_2_mbox_loc_grad"
  input: "LR"
  output: "conv6_2_mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_conf"
  input: "ONE"
  input: "fc7_mbox_conf_grad"
  input: "LR"
  output: "fc7_mbox_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc6_w"
  input: "ONE"
  input: "fc6_w_grad"
  input: "LR"
  output: "fc6_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_2_b"
  input: "ONE"
  input: "conv4_2_b_grad"
  input: "LR"
  output: "conv4_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_2_w"
  input: "ONE"
  input: "conv4_2_w_grad"
  input: "LR"
  output: "conv4_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_conf_flat"
  input: "ONE"
  input: "conv4_3_norm_mbox_conf_flat_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_conf_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc6_b"
  input: "ONE"
  input: "fc6_b_grad"
  input: "LR"
  output: "fc6_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_conf_b"
  input: "ONE"
  input: "fc7_mbox_conf_b_grad"
  input: "LR"
  output: "fc7_mbox_conf_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_loc"
  input: "ONE"
  input: "fc7_mbox_loc_grad"
  input: "LR"
  output: "fc7_mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_loc_w"
  input: "ONE"
  input: "conv9_2_mbox_loc_w_grad"
  input: "LR"
  output: "conv9_2_mbox_loc_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_1_w"
  input: "ONE"
  input: "conv8_1_w_grad"
  input: "LR"
  output: "conv8_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2"
  input: "ONE"
  input: "conv9_2_grad"
  input: "LR"
  output: "conv9_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_loc_perm"
  input: "ONE"
  input: "conv7_2_mbox_loc_perm_grad"
  input: "LR"
  output: "conv7_2_mbox_loc_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_loc_flat"
  input: "ONE"
  input: "conv4_3_norm_mbox_loc_flat_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_loc_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "SoftmaxWithLoss"
  input: "ONE"
  input: "SoftmaxWithLoss_autogen_grad"
  input: "LR"
  output: "SoftmaxWithLoss"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_1_b"
  input: "ONE"
  input: "conv5_1_b_grad"
  input: "LR"
  output: "conv5_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_3_w"
  input: "ONE"
  input: "conv5_3_w_grad"
  input: "LR"
  output: "conv5_3_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv1_1_b"
  input: "ONE"
  input: "conv1_1_b_grad"
  input: "LR"
  output: "conv1_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_loc_flat"
  input: "ONE"
  input: "fc7_mbox_loc_flat_grad"
  input: "LR"
  output: "fc7_mbox_loc_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_conf_perm"
  input: "ONE"
  input: "conv9_2_mbox_conf_perm_grad"
  input: "LR"
  output: "conv9_2_mbox_conf_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_conf_flat"
  input: "ONE"
  input: "conv9_2_mbox_conf_flat_grad"
  input: "LR"
  output: "conv9_2_mbox_conf_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2"
  input: "ONE"
  input: "conv6_2_grad"
  input: "LR"
  output: "conv6_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_1"
  input: "ONE"
  input: "conv6_1_grad"
  input: "LR"
  output: "conv6_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_1_w"
  input: "ONE"
  input: "conv5_1_w_grad"
  input: "LR"
  output: "conv5_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_conf_flat"
  input: "ONE"
  input: "conv6_2_mbox_conf_flat_grad"
  input: "LR"
  output: "conv6_2_mbox_conf_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_3_b"
  input: "ONE"
  input: "conv5_3_b_grad"
  input: "LR"
  output: "conv5_3_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_1"
  input: "ONE"
  input: "conv4_1_grad"
  input: "LR"
  output: "conv4_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_2"
  input: "ONE"
  input: "conv4_2_grad"
  input: "LR"
  output: "conv4_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3"
  input: "ONE"
  input: "conv4_3_grad"
  input: "LR"
  output: "conv4_3"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_conf_perm"
  input: "ONE"
  input: "fc7_mbox_conf_perm_grad"
  input: "LR"
  output: "fc7_mbox_conf_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv1_1_w"
  input: "ONE"
  input: "conv1_1_w_grad"
  input: "LR"
  output: "conv1_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm"
  input: "ONE"
  input: "conv4_3_norm_grad"
  input: "LR"
  output: "conv4_3_norm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv2_2"
  input: "ONE"
  input: "conv2_2_grad"
  input: "LR"
  output: "conv2_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv2_1"
  input: "ONE"
  input: "conv2_1_grad"
  input: "LR"
  output: "conv2_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_conf_b"
  input: "ONE"
  input: "conv8_2_mbox_conf_b_grad"
  input: "LR"
  output: "conv8_2_mbox_conf_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_b"
  input: "ONE"
  input: "conv9_2_b_grad"
  input: "LR"
  output: "conv9_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_loc_flat"
  input: "ONE"
  input: "conv9_2_mbox_loc_flat_grad"
  input: "LR"
  output: "conv9_2_mbox_loc_flat"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_1_b"
  input: "ONE"
  input: "conv4_1_b_grad"
  input: "LR"
  output: "conv4_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_loc_w"
  input: "ONE"
  input: "conv7_2_mbox_loc_w_grad"
  input: "LR"
  output: "conv7_2_mbox_loc_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_w"
  input: "ONE"
  input: "conv4_3_w_grad"
  input: "LR"
  output: "conv4_3_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "pool3"
  input: "ONE"
  input: "pool3_grad"
  input: "LR"
  output: "pool3"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_conf_w"
  input: "ONE"
  input: "conv8_2_mbox_conf_w_grad"
  input: "LR"
  output: "conv8_2_mbox_conf_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conf_pred"
  input: "ONE"
  input: "conf_pred_grad"
  input: "LR"
  output: "conf_pred"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_1"
  input: "ONE"
  input: "conv8_1_grad"
  input: "LR"
  output: "conv8_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2"
  input: "ONE"
  input: "conv8_2_grad"
  input: "LR"
  output: "conv8_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_1_w"
  input: "ONE"
  input: "conv4_1_w_grad"
  input: "LR"
  output: "conv4_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_b"
  input: "ONE"
  input: "conv4_3_b_grad"
  input: "LR"
  output: "conv4_3_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_loc_b"
  input: "ONE"
  input: "conv7_2_mbox_loc_b_grad"
  input: "LR"
  output: "conv7_2_mbox_loc_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_w"
  input: "ONE"
  input: "conv8_2_w_grad"
  input: "LR"
  output: "conv8_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_conf_b"
  input: "ONE"
  input: "conv7_2_mbox_conf_b_grad"
  input: "LR"
  output: "conv7_2_mbox_conf_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_b"
  input: "ONE"
  input: "conv8_2_b_grad"
  input: "LR"
  output: "conv8_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "pool5"
  input: "ONE"
  input: "pool5_grad"
  input: "LR"
  output: "pool5"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_mbox_conf_w"
  input: "ONE"
  input: "conv7_2_mbox_conf_w_grad"
  input: "LR"
  output: "conv7_2_mbox_conf_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_2_w"
  input: "ONE"
  input: "conv5_2_w_grad"
  input: "LR"
  output: "conv5_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc6"
  input: "ONE"
  input: "fc6_grad"
  input: "LR"
  output: "fc6"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7"
  input: "ONE"
  input: "fc7_grad"
  input: "LR"
  output: "fc7"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv1_2_b"
  input: "ONE"
  input: "conv1_2_b_grad"
  input: "LR"
  output: "conv1_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2"
  input: "ONE"
  input: "conv7_2_grad"
  input: "LR"
  output: "conv7_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_1"
  input: "ONE"
  input: "conv7_1_grad"
  input: "LR"
  output: "conv7_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_2_b"
  input: "ONE"
  input: "conv5_2_b_grad"
  input: "LR"
  output: "conv5_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_conf_b"
  input: "ONE"
  input: "conv4_3_norm_mbox_conf_b_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_conf_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv1_2_w"
  input: "ONE"
  input: "conv1_2_w_grad"
  input: "LR"
  output: "conv1_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "SmoothL1Loss"
  input: "ONE"
  input: "SmoothL1Loss_autogen_grad"
  input: "LR"
  output: "SmoothL1Loss"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_1"
  input: "ONE"
  input: "conv5_1_grad"
  input: "LR"
  output: "conv5_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_1_b"
  input: "ONE"
  input: "conv9_1_b_grad"
  input: "LR"
  output: "conv9_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_3"
  input: "ONE"
  input: "conv5_3_grad"
  input: "LR"
  output: "conv5_3"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv5_2"
  input: "ONE"
  input: "conv5_2_grad"
  input: "LR"
  output: "conv5_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_1_w"
  input: "ONE"
  input: "conv9_1_w_grad"
  input: "LR"
  output: "conv9_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_3"
  input: "ONE"
  input: "conv3_3_grad"
  input: "LR"
  output: "conv3_3"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_2"
  input: "ONE"
  input: "conv3_2_grad"
  input: "LR"
  output: "conv3_2"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_1"
  input: "ONE"
  input: "conv3_1_grad"
  input: "LR"
  output: "conv3_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_conf_perm"
  input: "ONE"
  input: "conv6_2_mbox_conf_perm_grad"
  input: "LR"
  output: "conv6_2_mbox_conf_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "loc_pred"
  input: "ONE"
  input: "loc_pred_grad"
  input: "LR"
  output: "loc_pred"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_loc_perm"
  input: "ONE"
  input: "conv9_2_mbox_loc_perm_grad"
  input: "LR"
  output: "conv9_2_mbox_loc_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv2_1_b"
  input: "ONE"
  input: "conv2_1_b_grad"
  input: "LR"
  output: "conv2_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_1"
  input: "ONE"
  input: "conv9_1_grad"
  input: "LR"
  output: "conv9_1"
  name: ""
  type: "WeightedSum"
}
op {
  input: "scale"
  input: "ONE"
  input: "scale_grad"
  input: "LR"
  output: "scale"
  name: ""
  type: "WeightedSum"
}
op {
  input: "fc7_mbox_loc_perm"
  input: "ONE"
  input: "fc7_mbox_loc_perm_grad"
  input: "LR"
  output: "fc7_mbox_loc_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv2_1_w"
  input: "ONE"
  input: "conv2_1_w_grad"
  input: "LR"
  output: "conv2_1_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "data"
  input: "ONE"
  input: "data_grad"
  input: "LR"
  output: "data"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_conf"
  input: "ONE"
  input: "conv6_2_mbox_conf_grad"
  input: "LR"
  output: "conv6_2_mbox_conf"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_2_mbox_loc_perm"
  input: "ONE"
  input: "conv8_2_mbox_loc_perm_grad"
  input: "LR"
  output: "conv8_2_mbox_loc_perm"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_2_w"
  input: "ONE"
  input: "conv3_2_w_grad"
  input: "LR"
  output: "conv3_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv9_2_mbox_loc"
  input: "ONE"
  input: "conv9_2_mbox_loc_grad"
  input: "LR"
  output: "conv9_2_mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_w"
  input: "ONE"
  input: "conv7_2_w_grad"
  input: "LR"
  output: "conv7_2_w"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv4_3_norm_mbox_loc"
  input: "ONE"
  input: "conv4_3_norm_mbox_loc_grad"
  input: "LR"
  output: "conv4_3_norm_mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv6_2_mbox_conf_b"
  input: "ONE"
  input: "conv6_2_mbox_conf_b_grad"
  input: "LR"
  output: "conv6_2_mbox_conf_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv3_2_b"
  input: "ONE"
  input: "conv3_2_b_grad"
  input: "LR"
  output: "conv3_2_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv8_1_b"
  input: "ONE"
  input: "conv8_1_b_grad"
  input: "LR"
  output: "conv8_1_b"
  name: ""
  type: "WeightedSum"
}
op {
  input: "mbox_loc"
  input: "ONE"
  input: "mbox_loc_grad"
  input: "LR"
  output: "mbox_loc"
  name: ""
  type: "WeightedSum"
}
op {
  input: "conv7_2_b"
  input: "ONE"
  input: "conv7_2_b_grad"
  input: "LR"
  output: "conv7_2_b"
  name: ""
  type: "WeightedSum"
}
external_input: "conv1_1_w"
external_input: "conv1_1_b"
external_input: "conv1_2_w"
external_input: "conv1_2_b"
external_input: "conv2_1_w"
external_input: "conv2_1_b"
external_input: "conv2_2_w"
external_input: "conv2_2_b"
external_input: "conv3_1_w"
external_input: "conv3_1_b"
external_input: "conv3_2_w"
external_input: "conv3_2_b"
external_input: "conv3_3_w"
external_input: "conv3_3_b"
external_input: "conv4_1_w"
external_input: "conv4_1_b"
external_input: "conv4_2_w"
external_input: "conv4_2_b"
external_input: "conv4_3_w"
external_input: "conv4_3_b"
external_input: "conv5_1_w"
external_input: "conv5_1_b"
external_input: "conv5_2_w"
external_input: "conv5_2_b"
external_input: "conv5_3_w"
external_input: "conv5_3_b"
external_input: "fc6_w"
external_input: "fc6_b"
external_input: "fc7_w"
external_input: "fc7_b"
external_input: "conv6_1_w"
external_input: "conv6_1_b"
external_input: "conv6_2_w"
external_input: "conv6_2_b"
external_input: "conv7_1_w"
external_input: "conv7_1_b"
external_input: "conv7_2_w"
external_input: "conv7_2_b"
external_input: "conv8_1_w"
external_input: "conv8_1_b"
external_input: "conv8_2_w"
external_input: "conv8_2_b"
external_input: "conv9_1_w"
external_input: "conv9_1_b"
external_input: "conv9_2_w"
external_input: "conv9_2_b"
external_input: "scale"
external_input: "conv4_3_norm_mbox_loc_w"
external_input: "conv4_3_norm_mbox_loc_b"
external_input: "conv4_3_norm_mbox_conf_w"
external_input: "conv4_3_norm_mbox_conf_b"
external_input: "fc7_mbox_loc_w"
external_input: "fc7_mbox_loc_b"
external_input: "fc7_mbox_conf_w"
external_input: "fc7_mbox_conf_b"
external_input: "conv6_2_mbox_loc_w"
external_input: "conv6_2_mbox_loc_b"
external_input: "conv6_2_mbox_conf_w"
external_input: "conv6_2_mbox_conf_b"
external_input: "conv7_2_mbox_loc_w"
external_input: "conv7_2_mbox_loc_b"
external_input: "conv7_2_mbox_conf_w"
external_input: "conv7_2_mbox_conf_b"
external_input: "conv8_2_mbox_loc_w"
external_input: "conv8_2_mbox_loc_b"
external_input: "conv8_2_mbox_conf_w"
external_input: "conv8_2_mbox_conf_b"
external_input: "conv9_2_mbox_loc_w"
external_input: "conv9_2_mbox_loc_b"
external_input: "conv9_2_mbox_conf_w"
external_input: "conv9_2_mbox_conf_b"
external_input: "ITER"
external_input: "ONE"
