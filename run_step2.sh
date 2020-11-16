save_dir="./results/MSNet_AD-anytime-cifar100-step=4-4-6-2-2-10-2"
pretrained_dir="./results/Anytime-cifar100-step=4-4-6-2-2-10-2"

python main_step2.py \
--data-root /data \
--save $save_dir \
--data cifar100 \
--gpu 0 \
--arch msnet \
--batch-size 128 \
--epochs 310 \
--lr-type SGDR \
--nBlocks 7 \
--step 4-4-6-2-2-10-2 \
--nChannels 12 \
--growthRate 6 \
--grFactor 1-2-4 \
--bnFactor 1-2-4 \
--transFactor 2-5 \
--pretrained $pretrained_dir/save_models/model_best.pth.tar \
-j 6