import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from deephd_losses import build_loss_by_name
from deephd_model import ASPPResNetSE
from pytorch_lightning import LightningModule
from new_data import CosMaPDataset, CosMaP_FCN5_Dataset
from new_data import is_cosmap
#
from new_model import ASPPResNetSE_mod
from segmentation_models_pytorch import Unet
#
from cosmap_unet import CosMaP_UNet
import cosmap_unet as cmun
#import additional_modules as add

#Фокусы с зерном рандомного генератора
def _get_random_seed():
    seed = int(time.time() * 100000) % 10000000 + os.getpid()
    return seed

#Тоже фокусы
def worker_init_fn_random(idx):
    seed_ = _get_random_seed() + idx
    torch.manual_seed(seed_)
    np.random.seed(seed_)

class CosMaPPipeline(LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.trn_dataset = CosMaPDataset(config['data']['train'],
                                         config['hyperparams']['crop'],
                                         config['net'],
                                         config['hyperparams']['channels'])
        self.val_dataset = CosMaPDataset(config['data']['valid'],
                                         config['hyperparams']['crop'],
                                         config['net'],
                                         config['hyperparams']['channels'])
        
        if self.config['net'] == 'unet':
            self.model = Unet(encoder_weights = None,
                              encoder_name = config['hyperparams']['encoder'],
                              encoder_depth = config['hyperparams']['num_stages'],
                              decoder_channels = [256, 128, 64, 32, 16][0:config['hyperparams']['num_stages']],
                              decoder_use_batchnorm = False,
                              classes = 2,
                              in_channels = self.trn_dataset._get_channels_count())
        elif self.config['net'] == 'cosmap_mod' or self.config['net'] == 'cosmap_mod2':
            self.model = ASPPResNetSE_mod(inp = self.trn_dataset._get_channels_count(),
                                          out = 1,
                                          num_stages = config['hyperparams']['num_stages'],
                                          freezed_stages = config['hyperparams']['freezed_stages'],
                                          batch_norm_at_begin = self.config['net'] != "cosmap_mod2").float()
        else:
            self.model = ASPPResNetSE(inp = self.trn_dataset._get_channels_count(),
                                      out = 1,
                                      num_stages = config['hyperparams']['num_stages'],
                                      freezed_stages = config['hyperparams']['freezed_stages']).float()                                     #self.config['net'] == 'cosmap02'
        if is_cosmap(self.config['net']) or self.config['net'] == 'cosmacvp' or self.config['net'] == 'cosmaprime' or self.config['net'] == 'unet':
            lossname = 'l2'
        elif self.config['net'] == 'dhd':
            lossname = 'bce'
        else:
            raise RuntimeError(f"wrong net {self.config['net']} [PIPE]")
        self.loss = build_loss_by_name(lossname)


    def build(self):
        self.trn_dataset.build()
        self.val_dataset.build()
        self.trn_loader = DataLoader(self.trn_dataset,
                                     num_workers = self.config['threads']['train_loader'],
                                     batch_size = self.config['hyperparams']['batch'],
                                     worker_init_fn = worker_init_fn_random)
        self.val_loader = DataLoader(self.val_dataset,
                                     num_workers = self.config['threads']['valid_loader'],
                                     batch_size = self.config['hyperparams']['batch_val'],
                                     worker_init_fn = worker_init_fn_random)
        self.timer = time.time()
        return self

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x = batch['inp']
        y_gt = batch['out']
        y_pr = self.model(x)
        loss = self.loss(y_pr, y_gt)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x = batch['inp']
        y_gt = batch['out']
        y_pr = self.model(x)
        loss = self.loss(y_pr, y_gt)
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        #avg = torch.stack(outputs).mean()
        print(f'\n\ttraining   {self.current_epoch} end\t\ttime = {time.time() - self.timer}')
        self.timer = time.time()

    #def validation_epoch_end(self, outputs):
        #avg = torch.stack(outputs).mean()
        #print(f'\n\n\tvalidation {self.current_epoch}')
        #self.timer = time.time()

    def configure_optimizers(self):
        ret = torch.optim.Adam(self.model.parameters(),
                               lr = self.config['hyperparams']['learning_rate'])
        return ret

    def train_dataloader(self):
        return self.trn_loader

    def val_dataloader(self):
        return self.val_loader



##########################
#PIPELINE FOR COSMAP_UNET#
##########################


class CosMaP_UNet_Pipeline(LightningModule):

    def __init__(self, config: dict): 
        super().__init__()
        self.config = config
        self.use_extractor = False #config['unet']['extractor']
        self.trn_dataset = CosMaPDataset(config['data']['train'],
            config['hyperparams']['crop'],
            'cosmap_unet', calc_t = self.use_extractor,
            add_cv_r = config['unet']['version'] != 0)
        self.val_dataset = CosMaPDataset(config['data']['valid'],
                                         config['hyperparams']['crop'],
                                         'cosmap_unet', calc_t = self.use_extractor,
                                         add_cv_r = config['unet']['version'] != 0)
        self.model = CosMaP_UNet(config['unet']['encoder'],
                                 config['unet']['bottleneck'],
                                 output = 2 if config['unet']['version'] == 0 else 3,
                                 ver = config['unet']['version']).float()
        #if self.use_extractor:
        #    self.extractor = cmun.CosMaP_Extractor(True)
        #    self.loss = cmun.TwoVectorLoss(False, False)
        #else:
        self.loss = torch.nn.MSELoss()

    def build(self):
        self.trn_dataset.build()
        self.val_dataset.build()
        self.trn_loader = DataLoader(self.trn_dataset,
                                     num_workers = self.config['threads']['train_loader'],
                                     batch_size = self.config['hyperparams']['batch'],
                                     worker_init_fn = worker_init_fn_random)
        self.val_loader = DataLoader(self.val_dataset,
                                     num_workers = self.config['threads']['valid_loader'],
                                     batch_size = self.config['hyperparams']['batch_val'],
                                     worker_init_fn = worker_init_fn_random)
        self.timer = time.time()
        return self

    def forward(self, cosma, amino1, amino2):
        cosma = torch.unsqueeze(cosma, dim = 1)
        return self.model(cosma, amino1, amino2)
    
    def training_step(self, batch, batch_nb): #Учить без экстрактора
        #X
        cosma = torch.unsqueeze(batch['inp']['cos_ma'], dim = 1)
        amino1 = batch['inp']['amino1']
        amino2 = batch['inp']['amino2']
        #Y [true]
        cosmapw_true = torch.unsqueeze(batch['out']['cos_ma_pw'], dim = 1)
        cv_true = torch.unsqueeze(batch['out']['cv'], dim = 1)
        if self.config['unet']['version'] == 0: 
            y_true = torch.cat([cosmapw_true, cv_true], dim = 1)
        else:
            cvr_true = torch.unsqueeze(batch['out']['cv_r'], dim = 1)
            y_true = torch.cat([cosmapw_true, cv_true, cvr_true], dim = 1)
        y_pred = self.model(cosma, amino1, amino2)
        loss = self.loss(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        #X
        cosma = torch.unsqueeze(batch['inp']['cos_ma'], dim = 1)
        amino1 = batch['inp']['amino1']
        amino2 = batch['inp']['amino2']
        #if self.use_extractor:
        #    t1 = batch['inp']['t1']
        #    t2 = batch['inp']['t2']
        #Y [true]
        cosmapw_true = torch.unsqueeze(batch['out']['cos_ma_pw'], dim = 1)
        cv_true = torch.unsqueeze(batch['out']['cv'], dim = 1)
        #y_true = torch.cat([cosmapw_true, cv_true], dim = 1)
        #if self.use_extractor:
        #    y_true = self.extractor(y_true, t1, t2)
        #Y [pred]
        y_pred = self.model(cosma, amino1, amino2)
        if self.config['unet']['version'] == 0: 
            y_true = torch.cat([cosmapw_true, cv_true], dim = 1)
        else:
            cvr_true = torch.unsqueeze(batch['out']['cv_r'], dim = 1)
            y_true = torch.cat([cosmapw_true, cv_true, cvr_true], dim = 1)
        #if self.use_extractor:
        #    y_pred = self.extractor(y_pred, t1, t2)
        #   angles = self.loss.get_angles(y_pred, y_true)
        #    errors = self.loss.get_errors(y_pred, y_true)
        #    self.log('valid_V1A', torch.mean(angles[0]), prog_bar=False, on_step=True, on_epoch=True)
        #    self.log('valid_CVA', torch.mean(angles[1]), prog_bar=False, on_step=True, on_epoch=True)
        #    self.log('valid_V1E', torch.mean(errors[0]), prog_bar=False, on_step=True, on_epoch=True)
        #    self.log('valid_CVE', torch.mean(errors[1]), prog_bar=False, on_step=True, on_epoch=True)
        loss = self.loss(y_pred, y_true)
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        #avg = torch.stack(outputs).mean()
        print(f'\n\ttraining   {self.current_epoch} end\t\ttime = {time.time() - self.timer}')
        self.timer = time.time()
    
    
    def configure_optimizers(self):
        ret = torch.optim.Adam(self.model.parameters(),
                               lr = self.config['hyperparams']['learning_rate'])
        return ret

    def train_dataloader(self):
        return self.trn_loader

    def val_dataloader(self):
        return self.val_loader

##########################
#PIPELINE FOR COSMAP_FCN5#
##########################

from cosmap_fcn import CosMaP_FCN5, CosMaP_FCN5_BN
#from cosmap_crc import CRC3

class CosMaP_FCN5_Pipeline(LightningModule): #+CRC3

    def __init__(self, config: dict): 
        super().__init__()
        self.config = config
        if config['data']['new']:
            self.trn_dataset = CosMaP_FCN5_Dataset(config['data']['train'],
                                                   config['hyperparams']['crop'],
                                                   config['data']['check_align'],
                config['mode'] == 'train' or config['mode'] == 'predict' and (config['inspect']['data'] == 'train' or config['inspect']['data'] == 'both'))
            self.val_dataset = CosMaP_FCN5_Dataset(config['data']['valid'],
                config['hyperparams']['crop'], config['data']['check_align'],
                config['mode'] == 'train' or config['mode'] == 'predict' and (config['inspect']['data'] == 'valid' or config['inspect']['data'] == 'both'))
        else:
            self.trn_dataset = CosMaPDataset(config['data']['train'],
                config['hyperparams']['crop'], 'cosmap_fcn5', add_cv_r = True)
            self.val_dataset = CosMaPDataset(config['data']['valid'],
                config['hyperparams']['crop'], 'cosmap_fcn5', add_cv_r = True)
        if config['net'] == 'cosmap_crc3':
            #self.model = CRC3(input=41).float()             #!!!!!
            raise Exception('<DEMO>')
        else:
            if not config['fcn']['reduced']: #!
                self.model = CosMaP_FCN5(config['fcn']['resnet'], out4 = config['fcn']['last']).float() if not config['fcn']['batchN'] \
                        else CosMaP_FCN5_BN(out4 = config['fcn']['last']).float()
            else:
                self.model = CosMaP_FCN5(config['fcn']['resnet'], out1 = 64, out2 = 128, out3 = 256, out4 = config['fcn']['last']).float() if not config['fcn']['batchN'] \
                        else CosMaP_FCN5_BN(out1 = 64, out2 = 128, out3 = 256, out4 = config['fcn']['last']).float()
            #
        # if config['fcn']['track_cl']:
        #     self.track_cl = True
        #     self.cl_loss = add.CLLoss(0.8, 0.2)
        # else:
        self.track_cl = False
        #
        # if config['fcn']['use_cl']:
        #     self.loss = add.CLLoss(0.8, 0.2)
        # else:
        self.loss = torch.nn.MSELoss()

    #P2^t * P1 !!!
    def forward(self, cosma, amino1, amino2):
        m, n = cosma.shape[2:4]
        #cosma = torch.unsqueeze(cosma, dim = 1)
        amino2 = amino2[:, 1:, :].unsqueeze(3).expand(-1, -1, -1, n)
        amino1 = amino1[:, 1:, :].unsqueeze(2).expand(-1, -1, m, -1)
        x = torch.cat([cosma, amino2, amino1], 1)
        return self.model(x)

    def build(self):
        self.trn_dataset.build()
        self.val_dataset.build()
        self.trn_loader = DataLoader(self.trn_dataset,
                                     num_workers = self.config['threads']['train_loader'],
                                     batch_size = self.config['hyperparams']['batch'],
                                     worker_init_fn = worker_init_fn_random)
        self.val_loader = DataLoader(self.val_dataset,
                                     num_workers = self.config['threads']['valid_loader'],
                                     batch_size = self.config['hyperparams']['batch_val'],
                                     worker_init_fn = worker_init_fn_random)
        self.timer = time.time()
        return self

    def configure_optimizers(self):
        ret = torch.optim.Adam(self.model.parameters(),
                               lr = self.config['hyperparams']['learning_rate'])
        return ret

    def train_dataloader(self):
        return self.trn_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_nb):
        #X
        cosma = torch.unsqueeze(batch['inp']['cos_ma'], dim = 1)
        amino1 = batch['inp']['amino1']
        amino2 = batch['inp']['amino2']
        #Y [true]
        cosmapw_true = torch.unsqueeze(batch['out']['cos_ma_pw'], dim = 1)
        cv_true = torch.unsqueeze(batch['out']['cv'], dim = 1)
        cvr_true = torch.unsqueeze(batch['out']['cv_r'], dim = 1)
        y_true = torch.cat([cosmapw_true, cv_true, cvr_true], dim = 1)
        y_pred = self(cosma, amino1, amino2)
        loss = self.loss(y_pred, y_true)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        #
        if self.track_cl:
            clloss = self.cl_loss(y_pred, y_true)
            self.log('cl_train_loss', clloss, prog_bar=False, on_step=True, on_epoch=True)
        #
        #self.model.log_weights(self.logger.experiment)
        return loss

    def validation_step(self, batch, batch_nb):
        #X
        cosma = torch.unsqueeze(batch['inp']['cos_ma'], dim = 1)
        amino1 = batch['inp']['amino1']
        amino2 = batch['inp']['amino2']
        #Y [true]
        cosmapw_true = torch.unsqueeze(batch['out']['cos_ma_pw'], dim = 1)
        cv_true = torch.unsqueeze(batch['out']['cv'], dim = 1)
        cvr_true = torch.unsqueeze(batch['out']['cv_r'], dim = 1)
        y_true = torch.cat([cosmapw_true, cv_true, cvr_true], dim = 1)
        y_pred = self(cosma, amino1, amino2)
        loss = self.loss(y_pred, y_true)
        self.log('valid_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        #
        if self.track_cl:
            clloss = self.cl_loss(y_pred, y_true)
            self.log('cl_valid_loss', clloss, prog_bar=False, on_step=True, on_epoch=True)
        #
        return loss

    def training_epoch_end(self, outputs):
        #avg = torch.stack(outputs).mean()
        print(f'\n\ttraining   {self.current_epoch} end\t\ttime = {time.time() - self.timer}')
        self.model.log_weights(self.logger.experiment)#, self.current_epoch)
        self.timer = time.time()