#!/usr/bin/env bash
cd ..
#!/usr/bin/env bash

pwd
NUM_PROC=2
normalization=$1
model=$2
CONFIG=configs/ViT/vit.yaml


if [ ! -d "output_test" ];then
    mkdir output_test
fi
output=output_test
chmod -R 777 ./output_test

python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=1235 train.py \
--data_train_root "/cfs/cfs-4260a4096/260a4096/445-mds3/xintaowang/datasets/imagenet/train" \
--data_train_label "/cfs/cfs-4260a4096/260a4096/public_datasets/imagenet/train_map.txt" \
--data_val_root "/cfs/cfs-4260a4096/260a4096/445-mds3/xintaowang/datasets/imagenet/val" \
--data_val_label "/cfs/cfs-4260a4096/260a4096/public_datasets/imagenet/val_map.txt" \
--config ${CONFIG} \
--port 1254 \
--batch-size 128 \
--norm_type $normalization \
--model $model \
--output $output \
