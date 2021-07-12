# single GPU training
python train.py -opt src/options/train/train_IRN_x4.yml


# distributed training
# # 4 GPUs
# nohup python -m torch.distributed.launch --nproc_per_node=3 --master_port=4321 train.py -opt options/train/train_IRN_x4.yml --launcher pytorch >out.log 2>&1 &


# fuser -v /dev/nvidia*
# ps -f -p 16323
# kill -9 681
