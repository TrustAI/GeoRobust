# nb_grid='1000' # 1d 2e3 2d 71^2 (5e3) 3d 23^3 (1.2e4) 4d 18^4 (1.04e5)
batch_size='128'
ag='0.111' # +- 20 degree
sft='0.1'
sc='0.1'
cuda='0'

model_list='
resnet50
'
# vit_base_patch16_224
# wide_resnet50_2
dataset_dir="/datasets/ImageNet2012/vaild" # update this

input="./imagenet_results/100_sample.txt"
ouput_dir="./imagenet_results/grid/"

for model in $model_list; do
# 1-D
    nb_grid='2000'
    ouput=${ouput_dir}${model}"_ag.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --scale $sc  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

# 2-D
    nb_grid='71'
    ouput=${ouput_dir}${model}"_sft.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --shift $sft  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_ag_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag --scale $sc  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

# 3-D
    nb_grid='23'
    ouput=${ouput_dir}${model}"_ag_sft.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag --shift $sft  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 

    done < "$input"

    ouput=${ouput_dir}${model}"_sft_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --shift $sft --scale $sc  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

# 4-D
    nb_grid='18'
    ouput=${ouput_dir}${model}"_ag_sft_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag --shift $sft --scale $sc  --model-name $model --grid-search --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"
done