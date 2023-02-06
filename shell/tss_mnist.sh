mnist_dir='/datasets/mnist-data' # update this

out_dir='../comparison_results/'

cuda='0'
root_model_dir='/datasets/TSS_model_weights/' # update this
model_name='checkpoint.pth.tar'

# MNIST
nb_iter='200'
nb_eval='3000'
depth='5'

#  rotation
output=${out_dir}"mnist_rotation.txt"
model_dir=${root_model_dir}'mnist/mnist_43/rotation-brightness/consistency/noise_0.12_r_55_b_0.2'
ag='0.27778' # +- 50 degree
line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --data-dir $mnist_dir --example-idx $idx --angle $ag --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model_name --model-dir $model_dir --cw --tss --po-set --po-set-size 2 >> $output
    line=$(( line + 1 ))
done

output=${out_dir}"mnist_rotation_grid.txt"
grid=2000
line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --grid-search --mnist --data-dir $mnist_dir --example-idx $idx --angle $ag --grid $grid --model-name $model_name --model-dir $model_dir --tss >> $output
    line=$(( line + 1 ))
done



##  scale
output=${out_dir}"mnist_scale.txt"
model_dir=${root_model_dir}'mnist/mnist_43/resize-brightness/consistency/noise_0.12_sl_0.45_sr_1.55_b_0.5/'
sc='0.3'
line=0
while [ "$line" -le 9999 ]; do
    CUDA_VISIBLE_DEVICES=$cuda python mnist_main.py --data-dir $mnist_dir --example-idx $line --scale $sc --max-iteration $nb_iter --max-deep $depth --max-evaluation $nb_eval --model-name $model_name --model-dir $model_dir --cw --tss --po-set --po-set-size 2 >> $output
    line=$(( line + 1 ))
done

output=${out_dir}"mnist_scale_grid.txt"
grid=2000
line=0
while [ "$line" -le 499 ]; do
    idx=$(( line * 20 ))
    CUDA_VISIBLE_DEVICES=$cuda python gridsearch_randompick.py --grid-search --mnist --data-dir $mnist_dir --example-idx $idx --scale $sc --grid $grid --model-name $model_name --model-dir $model_dir --tss >> $output
    line=$(( line + 1 ))
done

