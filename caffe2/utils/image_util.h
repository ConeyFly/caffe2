#ifndef CAFFE2_UTILS_IMAGE_UITLS_H_
#define CAFFE2_UTILS_IMAGE_UITLS_H_

#include "caffe/proto/caffe.pb.h"

namespace caffe2{



struct ResizeParam {
    bool resize_valid;
    float resize_prob;
    string resize_mode;
    int height;
    int width;
    int height_scale;
    int width_scale;
    vector<string> interp_mode;
};

struct EmitConstraintParam{
    bool emit_valid;
    string emit_type;
};

struct DistortParam{
    bool distort_valid;
    float brightness_prob;
    float brightness_delta;
    float contrast_prob;
    float contrast_lower;
    float contrast_upper;
    float hue_prob;
    float hue_delta;
    float saturation_prob;
    float saturation_lower;
    float saturation_upper;
    float random_order_prob;
};
struct ExpandParam{
    bool expand_valid;
    float expand_prob;
    float max_expand_ratio;
};

struct TransformParam{
    bool trans_valid;
    bool mirror;
    int crop;
    bool force_color;
    bool force_gray;
    vector<float> mean_value;
    ResizeParam resize_param;
    EmitConstraintParam emit_constraint;
    DistortParam distort_param;
    ExpandParam expand_param;
};


struct BatchSampler {
    int max_sample;
    int max_trials;
    float min_scale;
    float max_scale;
    float min_aspect_ratio;
    float max_aspect_ratio;
    float min_jaccard_overlap;
    bool use_original_image;
    float max_jaccard_overlap;
    float min_sample_coverage;
    float max_sample_coverage;
    float min_object_coverage;
    float max_object_coverage;
    BatchSampler(){max_jaccard_overlap=-1.0;
        min_sample_coverage=-1.0;
        max_sample_coverage=-1.0;
        min_object_coverage=-1.0;
        max_object_coverage=-1.0;}
    BatchSampler(int a,int b,float c,float d,float e,float f,float g,bool h):
    max_sample(a),max_trials(b),min_scale(c),max_scale(d),min_aspect_ratio(e),
    max_aspect_ratio(f),min_jaccard_overlap(g),use_original_image(h){max_jaccard_overlap=-1.0;
        min_sample_coverage=-1.0;
        max_sample_coverage=-1.0;
        min_object_coverage=-1.0;
        max_object_coverage=-1.0;}
};


struct AnnotatedDataParam {
    vector<BatchSampler> batch_sampler;
};

}

#endif