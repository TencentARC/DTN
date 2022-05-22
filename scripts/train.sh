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
--data_train_root "data train" \
--data_train_label "data train label" \
--data_val_root "data val" \
--data_val_label "data val label" \
--config ${CONFIG} \
--port 1254 \
--batch-size 128 \
--norm_type $normalization \
--model $model \
--output $output \
