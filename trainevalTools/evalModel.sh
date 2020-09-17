CUDA_VISIBLE_DEVICES=1 python3 train_hico.py --num-gpus 1 --config-file configs/CondInst/MS_R_101_3x.yaml \
--eval-only MODEL.WEIGHTS ./checkpoint/CondInst_new.pth