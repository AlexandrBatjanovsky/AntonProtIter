#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import torch
from torch import nn
from segmentation_models_pytorch.utils.losses import DiceLoss, JaccardLoss
#from torchmetrics.functional import precision, recall, matthews_corrcoef, pearson_corrcoef
#from segmentation_models_pytorch.utils.losses import BCEDiceLoss, BCEJaccardLoss, DiceLoss, JaccardLoss
#from lovasz import LovaszLoss


#Ну, всё ясно - лосс по строке
def build_loss_by_name(loss_name: str): #-> nn.Module:
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'l2': #
        return nn.MSELoss()
    # elif loss_name == 'bcedice':
    #     return BCEDiceLoss()
    # elif loss_name == 'bcejaccard':
    #     return BCEJaccardLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'jaccard':
        return JaccardLoss()
    #elif loss_name == 'mcc':
    #    return MatthewsL()
    #elif loss_name == 'lovasz':
    #    return LovaszLoss()
    #elif loss_name == 'prec':
    #    return PrecisionL()
    #elif loss_name == 'recc':
    #    return RecallL()
    #elif loss_name == 'corr':
    #    return CorrL()
    else:
        raise NotImplementedError
    #Huber!
    #Log!


def main_debug():
    y_pr = torch.rand([5, 5], dtype=torch.float32)
    y_gt = torch.randint(0, 2, [5, 5]).type(torch.float32)
    loss_names = ['bce','mcc']
    #loss_names = ['bce', 'bcedice', 'bcejaccard', 'dice', 'jaccard']

    loss_functions = [build_loss_by_name(x) for x in loss_names]
    loss_ = [f(y_pr, y_gt) for f in loss_functions]
    print(loss_)

    print('-')


if __name__ == '__main__':
    main_debug()