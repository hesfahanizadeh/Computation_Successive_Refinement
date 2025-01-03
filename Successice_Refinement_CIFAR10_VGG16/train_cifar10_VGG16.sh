python -u train_cifar10_VGG16.py \
    --comments 'training VGG16 on CIFAR10' \
    --device 'cuda:7' \
    \
    --dataset.dataset 'cifar10' \
    --dataset.path 'data' \
    \
    --randomseed 1 \
    --batch_size 128 \
    --num_epochs 300 \
    --learning_rate 0.005 \
