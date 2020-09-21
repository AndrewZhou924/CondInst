python3 train_hico.py --num-gpus 1 --config-file configs/CondInst/MS_R_101_3x.yaml \
--resume MODEL.WEIGHTS ./checkpoint/CondInst_new.pth \
SOLVER.IMS_PER_BATCH 2 SOLVER.MAX_ITER 2000000
