python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml rho 1 gamma 0
python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml rho 0.1 gamma 0
python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml rho 0.5 gamma 0
python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml rho 10 gamma 0
# # python3 train_stage1_vgg.py --cfg ./config/cifar10/cifar10_imb01_stage1_mixup_vgg.yaml
# # python3 train_stage1_vgg.py --cfg ./config/cifar10/cifar10_imb002_stage1_mixup_vgg.yaml
# # python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup.yaml
# # python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb01_stage1_mixup.yaml
# python3 train_stage1.py --cfg ./config/cifar10/cifar10_imb002_stage1_mixup.yaml