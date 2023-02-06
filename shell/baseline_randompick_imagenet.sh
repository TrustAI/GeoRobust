nb_grid='1000'
batch_size='128'
ag='0.111' # +- 30 degree
sft='0.1'
sc='0.1'
cuda='0'
model_list='
resnet50
'
dataset_dir="/datasets/ImageNet2012/vaild" # update this

input="./imagenet_results/100_sample.txt"
ouput_dir="./imagenet_results/random_pick/"

for model in $model_list; do
# 1-D
    ouput=${ouput_dir}${model}"_ag.txt"
    nb_pick='2000'
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --scale $sc  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

# 2-D
    ouput=${ouput_dir}${model}"_sft.txt"
    nb_pick='4000'
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --shift $sft  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_ag_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag --scale $sc  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

# 3-D
    ouput=${ouput_dir}${model}"_ag_sft.txt"
    nb_pick='6000'
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag --shift $sft  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_sft_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --shift $sft --scale $sc  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"

# 4-D
    ouput=${ouput_dir}${model}"_ag_sft_sc.txt"
    nb_pick='8000'
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --example-idx $line --angle $ag --shift $sft --scale $sc  --model-name $model --random-pick --pick-size $nb_pick --grid $nb_grid --batch-size $batch_size --imagenet --data-dir $dataset_dir >> $ouput 
    done < "$input"
done