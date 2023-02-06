nb_iter='150'
nb_eval='2000'
depth='6'
ag='0.111'
sft='0.1'
sc='0.1'
cuda='0'

model_list='
resnet50
'

dataset_dir="/datasets/ImageNet2012/vaild" # update this
input="./imagenet_results/100_sample.txt"
ouput_dir="./imagenet_results/GeoRobust/"

for model in $model_list; do

    ouput=${ouput_dir}${model}"_ag.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --angle $ag --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_sft.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --shift $sft --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_ag_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --angle $ag --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_ag_sft.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --angle $ag --shift $sft --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_sft_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --shift $sft --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

    ouput=${ouput_dir}${model}"_ag_sft_sc.txt"
    while IFS= read -r line
    do
        CUDA_VISIBLE_DEVICES=$cuda python imagenet_main.py --example-idx $line --angle $ag --shift $sft --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model --timm-model --data-dir $dataset_dir --po-set --po-set-size 2 >> $ouput 
    done < "$input"

done