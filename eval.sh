python3 eval.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup.yaml resume ./saved/cifar10_imb001_cbl_stage1_mixup_2/ckps/model_best.pth.tar
python3 eval_lda.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup.yaml resume ./saved/cifar10_imb001_cbl_stage1_mixup_2/ckps/model_best.pth.tar
python3 eval_lda_mixup.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup.yaml resume ./saved/cifar10_imb001_cbl_stage1_mixup_2/ckps/model_best.pth.tar
# python3 eval.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml resume ./saved/vgg_convex_loss_mixup/ckps/model_best.pth.tar
# python3 eval_lda.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml resume ./saved/vgg_convex_loss_mixup/ckps/model_best.pth.tar
# python3 eval_lda_mixup.py --cfg ./config/cifar10/cifar10_imb001_stage1_mixup_vgg.yaml resume ./saved/vgg_convex_loss_mixup/ckps/model_best.pth.tar
