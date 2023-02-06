nb_iter='150'
nb_eval='3000'
depth='6'
ag='0.111'
sft='0.1'
sc='0.1'
cuda='0'
dataset_dir="/datasets/ImageNet2012/vaild" # update this

# --- model list ---#

# resnet50
# resnet152
# wide_resnet50_2
# wide_resnet101_2
# adv_inception_v3
# inception_v3
# inception_v4
# xcit_medium_24_p16_224
# beit_base_patch16_224
# vit_base_patch16_224
# mixer_b16_224
# gmlp_s16_224
# swin_base_patch4_window7_224
# pit_b_224
# vit_base_patch32_224
# vit_large_patch16_224
# vit_large_patch32_224
# resnet34
# resnet101
# beit_large_patch16_224
# -------------------#

model_list=' 
resnet50
'

input="./imagenet_results/500_sample.txt"
ouput_dir="./imagenet_results/benchmark/"

for model in $model_list; do

    ouput=${ouput_dir}${model}"_ag_sft_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --angle $ag --shift $sft --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --data-dir $dataset_dir --timm-model >> $ouput 
    done < "$input"

done