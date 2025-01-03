python -u train_cifar10_VGG16.py \
    --comments 'training VGG16 on CIFAR10' \
    --device 'cuda:7' \
    \
    --eval_job '2024-06-17 16:39:47' \
    --dataset.dataset 'cifar10' \
    --dataset.path 'data' \
    \
    --randomseed 1 \
    --batch_size 128 \
