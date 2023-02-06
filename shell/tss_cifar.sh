cifar_dir='/datasets/cifar-data' # update this
out_dir='../comparison_results/'

cuda='0'
model_name='checkpoint.pth.tar'

root_model_dir='/datasets/TSS_model_weights/' # update this

# CIFAR10
nb_iter='200'
nb_eval='3000'
depth='5'

##  rotation 10
output=${out_dir}"cifar10_rotation_10.txt"
model_dir=${root_model_dir}'cifar10/resnet110/rotation-brightness/consistency/noise_0.05_r_12.5_b_0.2/' # update this
ag='0.05556' # +- 10 degree

line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python cifar_main.py --data-dir $cifar_dir --example-idx $idx --angle $ag --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model_name --model-dir $model_dir --cw --tss --po-set --po-set-size 2 >> $output
    line=$(( line + 1 ))
done

output=${out_dir}"cifar10_rotation_10_grid.txt"
grid=2000
line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --grid-search --cifar10 --data-dir $cifar_dir --example-idx $idx --angle $ag --grid $grid --model-name $model_name --model-dir $model_dir --tss >> $output
    line=$(( line + 1 ))
done


##  rotation 30
output=${out_dir}"cifar10_rotation_30.txt"
model_dir=${root_model_dir}'cifar10/resnet110/rotation-brightness/consistency/noise_0.05_r_35_b_0.2/' # update this
ag='0.16667' # +- 30 degree

line=0
while [ "$line" -le 9999 ]; do
    CUDA_VISIBLE_DEVICES=$cuda python cifar_main.py --data-dir $cifar_dir --example-idx $line --angle $ag --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model_name --model-dir $model_dir --cw --tss --po-set --po-set-size 2 >> $output
    line=$(( line + 1 ))
done

output=${out_dir}"cifar10_rotation_30_grid.txt"
grid=2000
line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --grid-search --cifar10 --data-dir $cifar_dir --example-idx $idx --angle $ag --grid $grid --model-name $model_name --model-dir $model_dir --tss >> $output
    line=$(( line + 1 ))
done

##  scale
output=${out_dir}"cifar10_scale.txt"
model_dir=${root_model_dir}'cifar10/resnet110/resize-brightness/consistency/noise_0.12_sl_0.65_sr_1.35_b_0.3/' # update this
sc='0.3'
line=0
while [ "$line" -le 9999 ]; do
    CUDA_VISIBLE_DEVICES=$cuda python cifar_main.py --data-dir $cifar_dir --example-idx $line --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model_name --model-dir $model_dir --cw --tss --po-set --po-set-size 2 >> $output
    line=$(( line + 1 ))
done

output=${out_dir}"cifar10_scale_grid.txt"
grid=2000
line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --grid-search --cifar10 --data-dir $cifar_dir --example-idx $idx --scale $sc --grid $grid --model-name $model_name --model-dir $model_dir --tss >> $output
    line=$(( line + 1 ))
done