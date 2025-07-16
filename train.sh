export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
export WANDB_KEY=f9831e23517e27f7ecac9b54bc2cdcabb3af8c33
export ENTITY=diffusion-guidance
export PROJECT=sit-cfg-distill
# python train.py --model SiT-XL/2 --data-path /media/dataset1/ImageNet2012/train --ckpt /media/dataset2/jiwon/representation/SiT/checkpoint/SiT-XL-2-256.pt --wandb 
python -m torch.distributed.run  --nnodes=1 --nproc_per_node=1 train.py --model SiT-XL/2 --data-path /media/dataset1/ImageNet2012/train --ckpt /media/dataset2/jiwon/representation/SiT/checkpoint/SiT-XL-2-256.pt --wandb --global-batch-size 32