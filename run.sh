# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml batch_size 64 lr 0.01 weight_decay 5e-3 name cifar100_imb100
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml batch_size 64 lr 0.01 weight_decay 5e-3 name cifar100_imb100_instance
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml batch_size 64 lr 0.01 weight_decay 5e-3 name cifar100_imb100_cos cos True 
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml batch_size 128 weight_decay 1e-3 name cifar100_imb100_lr1e-3
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml batch_size 128 weight_decay 5e-3 name cifar100_imb100_lr5e-3
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml name cifar100_normal
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml name cifar100_imb100_normal_cos cos True
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml weight_decay 5e-4 name cifar100_imb100_lr5e-4
python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml weight_decay 7e-4 name cifar100_imb100_lr7e-4_mm
# python3 train_stage1.py --cfg ./config/cifar100/cifar100_imb001_stage1_mixup.yaml weight_decay 9e-4 name cifar100_imb100_lr9e-4


