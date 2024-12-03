#!/bin/bash

input_dir=path/to/input_images
output_dir=path/to/output_images

conf="conf/conditional_continuous_linear_df8kost_dim128.yaml"
model="models/srgd/conditional_continuous_linear_df8kost_dim128_epoch300.pth"
test_label=0
class_cond_scale=1.0
seed=71

python inference.py -c ${conf} -m ${model} \
  --class_cond_scale ${class_cond_scale} --test_label ${test_label} --seed ${seed} \
  --input_dir ${input_dir} --output_dir ${output_dir}
