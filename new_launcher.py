#!/home/alexandersn/anaconda3/bin/python3
# -*- coding: utf-8 -*-

import os
import time
import json
import torch
import random
import pandas as pd
import numpy as np
import calculus as clc
import scipy.special as spec
import new_data as ndata
#import input_output as io
import new_preproc as nproc
import new_pipeline as npipe
import matplotlib.pyplot as plt
from task_utils import parallel_tasks_run_def
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys

#from cosmap_unet import CosMaP_UNet
#from additional_modules import CLLoss

#Класс, для автоматизированной работы с сетью
#конфигурируется на основе json файла
#учтите, что файл конфигурации в результате работы программы будет изменён
class CosMaP:
    
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
            print(self.config)
            sys.exit()
    
    #Метод для массового перегона pdb в pkl
    # (вызывается автоматически по результатам парсинга конфигурационного файла)
    def preproc(self):
        print('================\nPreprocessing start')
        train_size = self.config['preproc']['train%']
        valid_size = 1 - train_size
        random_state = self.config['preproc']['random']
        pdb_dir = self.config['preproc']['pdb_dir']
        pkl_dir = self.config['preproc']['pkl_dir']
        print('Extracting PDBs...')
        pdb_idx = pd.read_csv(self.config['preproc']['pdb_list'], sep=';')
        pdb_paths = [os.path.join(pdb_dir, x) for x in pdb_idx['path']]
        pdb_len = len(pdb_paths)
        print(f'PDBs count = {pdb_len}')
        task_data = [[x, pkl_dir, xi, pdb_len] for xi, x in enumerate(pdb_paths)] #default params
        print('Producing PKLs...')
        ret = parallel_tasks_run_def(nproc.read_and_dump, task_data,
                                     num_workers = self.config['threads']['preproc'],
                                     use_process = self.config['threads']['use_process'])
        #whatever (should be faster)
        indx = []
        paths = []
        crops = []
        k = 0
        for x in ret:
            if x != None:
                indx.append(k)
                paths.append(x[0])
                crops.append(x[1])
                k += 1
        #
        print(f'PKLs count = {k}')
        print('Splitting...')
        indx.sort() #!
        split = train_test_split(indx, test_size = valid_size, train_size = train_size,
                                 random_state = random_state)
        train_indx = split[0]
        valid_indx = split[1]
        print(f'Train size = {len(train_indx)}')
        print(f'Test  size = {len(valid_indx)}')
        print('Extracting crops...')
        train = [paths[i] for i in train_indx]
        valid = [paths[i] for i in valid_indx]
        train_crops = [crops[i] for i in train_indx]
        valid_crops = [crops[i] for i in valid_indx]
        print('Saving paths...')
        train_path = os.path.join(pkl_dir, 'train_set.csv')
        valid_path = os.path.join(pkl_dir, 'valid_set.csv')
        train_frame = pd.DataFrame({'path': train, 'crop': train_crops})
        train_frame.to_csv(train_path, sep=';')
        valid_frame = pd.DataFrame({'path': valid, 'crop': valid_crops})
        valid_frame.to_csv(valid_path, sep=';')
        print(f"Train path = '{train_path}'")
        print(f"Valid path = '{valid_path}'")
        print('Preproccesing end\n================')
        return {'train_path': train_path, 'valid_path': valid_path}

    #Загрузчик пайплайна
    def load_pipe(self):
        checkpoint = self.config['attempt']['checkpoint']
        if checkpoint is None:
            print('No checkpoint')
            if self.config['net'] == 'cosmap_unet':
                pipeline = npipe.CosMaP_UNet_Pipeline(self.config)
            elif self.config['net'] == 'cosmap_fcn5' or self.config['net'] == 'cosmap_crc3':
                pipeline = npipe.CosMaP_FCN5_Pipeline(self.config)
            else:
                pipeline = npipe.CosMaPPipeline(self.config)
        else:
            print('Loading checkpoint...')
            if self.config['net'] == 'cosmap_unet':
                pipeline = npipe.CosMaP_UNet_Pipeline.load_from_checkpoint(checkpoint,
                                                                           config = self.config)
            elif self.config['net'] == 'cosmap_fcn5' or self.config['net'] == 'cosmap_crc3':
                pipeline = npipe.CosMaP_FCN5_Pipeline.load_from_checkpoint(checkpoint,
                                                                           config = self.config)
            else:
                pipeline = npipe.CosMaPPipeline.load_from_checkpoint(checkpoint,
                                                                     config = self.config)
        return pipeline

    #Пути для всяких сохранений (оч. удобно)
    def get_savings_paths(self):
        saves_dir = self.config['attempt']['saves_dir']
        if not os.path.exists(saves_dir):
            os.mkdir(saves_dir)
        tag = self.config['attempt']['tag']
        if tag is None:
            tag = self.config['net']
        attempt_name = f"{tag}_at{self.config['attempt']['attempt']}/"
        attempt_dir = os.path.join(saves_dir, attempt_name)
        if not os.path.exists(attempt_dir):
            os.mkdir(attempt_dir)
        save_name = 'save.ckpt' #used to have wrong extension 'cpkt'!!! Be aware.
        checkpoint_path = os.path.join(attempt_dir, save_name)
        config_path = os.path.join(attempt_dir, 'config.json')
        return {'attempt_name': attempt_name, 'attempt_dir': attempt_dir, 'save_name': save_name,
                'checkpoint_path': checkpoint_path, 'config_path': config_path}

    #Метод, вызываемый для обучения сети
    def train(self):
        print('================\nTraining start')
        pipeline = self.load_pipe()
        pipeline.build()
        paths = self.get_savings_paths()
        #
        epochs = self.config['attempt']['attempt_epochs']
        checkpoint = self.config['attempt']['checkpoint']
        resuming = not checkpoint is None
        if resuming:
            print('Resume training...')
            epochs += self.config['attempt']['epoch'] #wrong!!!!!
        #
        with open(paths['config_path'], 'w') as config_file:
            json.dump(self.config, config_file, indent = 4)
        callback = ModelCheckpoint(dirpath = paths['attempt_dir'], mode = 'min')
        logger = TensorBoardLogger(save_dir = paths['attempt_dir'], name = '', version = 0)
        trainer = Trainer(accelerator = self.config['attempt']['accelerator'],
                          callbacks = [callback],
                          max_epochs = epochs,
                          logger = logger)
        print('Training...')
        #
        if resuming:
            trainer.fit(pipeline, ckpt_path=checkpoint)
        else: 
            trainer.fit(pipeline)
        #
        print('Saving results...')
        trainer.save_checkpoint(paths['checkpoint_path'])
        print('Training end\n================')
        return {'new_checkpoint': paths['checkpoint_path'],
                'new_attempt': self.config['attempt']['attempt'] + 1,
                'new_epoch': self.config['attempt']['epoch'] + self.config['attempt']['attempt_epochs']}

    #Взять случайный белок на обследование
    def get_item(self, pipeline):
        if self.config['inspect']['data'] == 'both':
            name = random.sample(['train', 'valid'], k = 1)[0]
        else:
            name = self.config['inspect']['data']
        if name == 'train':
            dataset = pipeline.trn_dataset
        elif name == 'valid':
            dataset = pipeline.val_dataset
        else:
            raise RuntimeError(f"Wrong dataset {name} [LAUNCHER]")
        index = random.randrange(len(dataset))
        return {'dataset': name, 'index': index,
                'item': dataset._justload(index),
                'input': dataset._getitem_full(index)}

    #Визуализация для dhd
    def dhd_visualize(self, protein, i, pipeline):
        print(f"[{protein['dataset']}] {protein['item']['prot_name']} ({i}) [DHD]")
        with torch.no_grad():
            result0 = result = pipeline(torch.tensor(np.expand_dims(protein['input']['inp'], axis = 0)).cuda()).detach().cpu().numpy()[0]
        result = spec.expit(result)
        inter = result > 0.5
        tinter = protein['item']['inter'][0]
        print(f"\tcorr = {np.corrcoef(np.ravel(result), np.ravel(tinter))[0, 1]}")
        print('Дискректизация по границе 0.5:')
        print(f"\tcorr = {np.corrcoef(np.ravel(inter), np.ravel(tinter))[0, 1]}")
        plt.subplot(2, 2, 1)
        plt.imshow(protein['item']['cos_ma'][0])
        #result0 = np.ravel(result0)
        #plt.hist([result0[np.logical_and(np.ravel(tinter == 0), result0 > -4)],
        #          result0[np.logical_and(np.ravel(tinter == 1), result0 > -4)]], histtype='step', bins=100)
        plt.subplot(2, 2, 4)
        plt.imshow(result)
        plt.subplot(2, 2, 3)
        plt.imshow(tinter)
        plt.subplot(2, 2, 2)
        plt.imshow(inter)
        plt.show()
        print('~~~~~~~~~~~~~~~~~~~~~~~~~')

    #Рисовательная рутинка (n x 3)
    def draw_prot(self, ax, prot1, prot2, color1 = 'blue', color2 = 'red', s = np.identity(3)):
        prot1 = prot1 @ s
        prot2 = prot2 @ s
        prot1 = np.column_stack([prot1[:, 1], prot1[:, 2], prot1[:, 0]])
        prot2 = np.column_stack([prot2[:, 1], prot2[:, 2], prot2[:, 0]])
        ax.scatter(prot1[:, 0], prot1[:, 1], prot1[:, 2], color = color1, s = 5)
        ax.scatter(prot2[:, 0], prot2[:, 1], prot2[:, 2], color = color2, s = 5)
        ax.plot(prot1[:, 0], prot1[:, 1], prot1[:, 2], color = 'black', linewidth = 0.5)
        ax.plot(prot2[:, 0], prot2[:, 1], prot2[:, 2], color = 'black', linewidth = 0.5)

    def cosmap_fcn5_visualize(self, protein, i, pipeline):
        n = protein['input']['inp']['cos_ma'].shape[0]
        print(f"[{protein['dataset']}] {protein['item']['prot_name']} ({i}) [COSMAP FCN5] <{n}>")
        with torch.no_grad():
            cosma = torch.tensor(np.expand_dims(protein['input']['inp']['cos_ma'], axis = (0, 1))).cuda()
            amino1 = torch.tensor(np.expand_dims(protein['input']['inp']['amino1'], axis = 0)).cuda()
            amino2 = torch.tensor(np.expand_dims(protein['input']['inp']['amino2'], axis = 0)).cuda()
            result = pipeline(cosma, amino1, amino2).detach().cpu().numpy()[0]
        #
        err1 = np.abs(protein['input']['out']['cos_ma_pw'] - result[0])
        err2 = np.abs(protein['input']['out']['cv'] - result[1])
        err3 = np.abs(protein['input']['out']['cv_r'] - result[2])
        plt.subplot(3, 4, 1)
        plt.imshow(protein['input']['inp']['cos_ma'], cmap = "gray")
        plt.subplot(3, 4, 2)
        plt.imshow(protein['input']['out']['cos_ma_pw'], cmap = "gray")
        plt.subplot(3, 4, 3)
        plt.imshow(protein['input']['out']['cv'], cmap = "gray")
        plt.subplot(3, 4, 4)
        plt.imshow(protein['input']['out']['cv_r'], cmap = "gray")
        #cv = (result[1] + np.transpose(result[2])) / 2
        #
        vecs = np.transpose(protein['item']['coords'][0])
        cv_extr = clc.cv_extractor(vecs, result[1], result[2])
        cv = cv_extr['cv']
        #
        err4 = np.abs(protein['input']['out']['cv'] - cv)
        #plt.subplot(3, 4, 5)
        #plt.imshow(cv)
        plt.subplot(3, 4, 5)
        plt.imshow(result[0], cmap = "gray")
        plt.subplot(3, 4, 7)
        plt.imshow(result[1], cmap = "gray")
        plt.subplot(3, 4, 8)
        plt.imshow(result[2], cmap = "gray")
        #plt.subplot(3, 4, 9)
        #plt.imshow(err4)
        plt.subplot(3, 4, 9)
        plt.imshow(err1, cmap = "gray")
        plt.subplot(3, 4, 11)
        #plt.imshow(err2)
        plt.imshow(cv, cmap = "gray")
        plt.subplot(3, 4, 12)
        #plt.imshow(err3)
        plt.imshow(np.abs(cv - result[1]), cmap = "gray")
        #plt.figure()
        #plt.hist([np.ravel(err1), np.ravel(err2), np.ravel(err3), np.ravel(err4)], histtype = 'step', bins = 100)
        
        true_stat = clc.fast_q(vecs, protein['input']['out']['cos_ma_pw'])
        pred_stat = clc.fast_q(vecs, result[0])
        print('\nМатрица поворота Q:')
        print(f"\tL1 = {np.mean(err1)}")
        print(f"\tL2 = {np.sqrt(np.mean(err1 ** 2))}")
        print(f"\t||Q_tr - Q_pr|| = {np.linalg.norm(true_stat['Q_ref'] - pred_stat['Q_ref'])}")
        angl = np.arccos(np.abs(np.transpose(true_stat['vec1']) @ pred_stat['vec1']))[0, 0]
        print(f"\tУгловая ошибка = {angl * 180 / np.pi} degrees")
        true_s = np.transpose(true_stat['S'])[:,(2,1,0)]
        pred_s = np.transpose(pred_stat['S'])[:,(2,1,0)]
        q = pred_stat['Q_ref']
        vec1 = pred_stat['vec1']
        #error estimation
        ests = clc.cosma_var(vecs, q, result[0])
        #print(f"\tl1 = {ests['abs']}")
        print(f"\n\tДисперсия Q = {ests['var']}")
        plt.subplot(3, 4, 6)
        plt.imshow(ests['cosma'], cmap = "gray")
        plt.subplot(3, 4, 10)
        plt.imshow(np.abs(ests['err']), cmap = "gray")
        #
        print('\nЦентральный вектор CV:')
        print(f"\tL1 = {np.mean(err4)}")
        print(f"\tL2 = {np.sqrt(np.mean(err4 ** 2))}")
        true_stat = clc.left_cv(vecs, protein['input']['out']['cv'])
        #pred_stat = clc.left_cv(vecs, cv)
        pred_stat = cv_extr
        angl = np.arccos(np.transpose(true_stat['center_vec']) @ pred_stat['center_vec'])[0, 0]
        print(f"\tИсходная угловая ошибка = {angl * 180 / np.pi} degrees")
        ref_stat = clc.ortagonalize(vec1, pred_stat['center_vec'])
        angl = np.arccos(np.transpose(true_stat['center_vec']) @ ref_stat['center_vec'])[0, 0]
        print(f"\tУО после согласования с Q = {angl * 180 / np.pi} degrees")
        cv = ref_stat['center_vec']

        print(f"\n\tДисперсия CV = {pred_stat['dev']**2}")
        #
        #NAIVE BUMPING
        #
        print('\nBUMP:')
        coord = np.transpose(protein['item']['coords'][0])
        coord_true = np.transpose(protein['item']['coords'][1])
        coord_pred = clc.rotate_prot(coord, q)
        shift = clc.find_first0(coord, coord_pred, self.config['bumping']['distance'], cv, self.config['bumping']['tolerance'])
        print(f"\tf = {shift['f']}")
        print(f"\tshift = {shift['shift']} A*")
        coord_pred = clc.shift_prot(coord_pred, cv, shift['shift'])
        print(f"\tRMSD = {clc.squared_mean_error(coord_pred, coord_true)} A*")
        #
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection = '3d')
        ax.set_box_aspect((1, 1, 1))
        coord = np.transpose(coord)
        coord_true = np.transpose(coord_true)
        coord_pred = np.transpose(coord_pred)
        self.draw_prot(ax, coord, coord_true, s = true_s)
        ax = fig.add_subplot(1, 2, 2, projection = '3d')
        ax.set_box_aspect((1, 1, 1))
        self.draw_prot(ax, coord, coord_pred, color2 = 'purple', s = pred_s)
        #
        print('~~~~~~~~~~~~~~~~~~~~~~~~~')
        plt.show()

    #Метод для исследования результатов обучения
    def inspect(self):
        print('================\nInspection start')
        pipeline = self.load_pipe()
        pipeline.build().cuda()
        if self.config['data']['new']:
            for i in range(self.config['inspect']['count']):
                try:
                    self.new_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [NEW DATASET]')
        elif npipe.is_cosmap(self.config['net']):
            for i in range(self.config['inspect']['count']):
                try:
                    self.cosmap_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [COSMAP]')
        elif self.config['net'] == 'dhd':
            for i in range(self.config['inspect']['count']):
                try:
                    self.dhd_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [DHD]')
        elif self.config['net'] == 'cosmacvp':
            for i in range(self.config['inspect']['count']):
                try:
                    self.cosmacvp_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [COSMAcvP]')
        elif self.config['net'] == 'cosmaprime':
            for i in range(self.config['inspect']['count']):
                try:
                    self.cosmaprime_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [COSMAPrime]')
        elif self.config['net'] == 'unet':
            for i in range(self.config['inspect']['count']):
                try:
                    self.unet_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [UNET]')
        elif self.config['net'] == 'cosmap_unet':
            for i in range(self.config['inspect']['count']):
                try:
                    self.cosmap_unet_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [COSMAP UNET]')
        elif self.config['net'] == 'cosmap_fcn5' or self.config['net'] == 'cosmap_crc3':
            for i in range(self.config['inspect']['count']):
                try:
                    self.cosmap_fcn5_visualize(self.get_item(pipeline), i, pipeline)
                except:
                    print('ERROR [COSMAP FCN5]')
        else:
            print('Inspection of this net is not realised.')
        print('================\nInspection end')

    #Итерация предсказания для CosMaP FCN5
    def predict_cosmap_fcn5_iteration(self, protein, i, pipeline):
        size = protein['input']['inp']['cos_ma'].shape[0]
        #timer = time.time()
        try:
            with torch.no_grad():
                cosma = torch.tensor(np.expand_dims(protein['input']['inp']['cos_ma'], axis = (0,1))).cuda()
                amino1 = torch.tensor(np.expand_dims(protein['input']['inp']['amino1'], axis = 0)).cuda()
                amino2 = torch.tensor(np.expand_dims(protein['input']['inp']['amino2'], axis = 0)).cuda()
                result = pipeline(cosma, amino1, amino2).detach().cpu().numpy()[0]
            cl = clc.cl_loss(0.8, 0.2, result, np.array([protein['input']['out']['cos_ma_pw'],
                                                         protein['input']['out']['cv'],
                                                         protein['input']['out']['cv_r']]))['loss']
            #result[1] = (result[1] + np.transpose(result[2])) / 2 #merge
            #
            err_q = np.abs(protein['input']['out']['cos_ma_pw'] - result[0])
            err_cv = np.abs(protein['input']['out']['cv'] - result[1])
            #
            err_q_l1 = np.mean(err_q)
            err_q_l2 = np.sqrt(np.mean(err_q ** 2))
            err_cv_l1 = np.mean(err_cv)
            err_cv_l2 = np.sqrt(np.mean(err_cv ** 2))
            #
            coord = np.transpose(protein['item']['coords'][0])
            true_q = clc.fast_q(coord, protein['input']['out']['cos_ma_pw'])
            pred_q = clc.fast_q(coord, result[0])
            qvar = clc.cosma_var(coord, pred_q['Q_ref'], result[0]) #NEW
            norm_err = np.linalg.norm(true_q['Q_ref'] - pred_q['Q_ref'])
            v1_angl = np.arccos(np.abs(np.transpose(true_q['vec1']) @ pred_q['vec1']))[0, 0] * 180 / np.pi
            #
            true_cv = clc.left_cv(coord, protein['input']['out']['cv'])
            #pred_cv = clc.left_cv(coord, result[1])
            pred_cv = clc.cv_extractor(coord, result[1], result[2])
            ref_cv = clc.ortagonalize(pred_q['vec1'], pred_cv['center_vec'])
            #
            cv0_angl = np.arccos(np.transpose(true_cv['center_vec']) @ pred_cv['center_vec'])[0, 0] * 180 / np.pi
            cv1_angl = np.arccos(np.transpose(true_cv['center_vec']) @ ref_cv['center_vec'])[0, 0] * 180 / np.pi
            #
            coord_true = np.transpose(protein['item']['coords'][1])
            coord_pred = clc.rotate_prot(coord, pred_q['Q_ref'])
            shift = clc.find_first0(coord, coord_pred, self.config['bumping']['distance'], ref_cv['center_vec'], self.config['bumping']['tolerance'])
            coord_pred = clc.shift_prot(coord_pred, ref_cv['center_vec'], shift['shift'])
            rmsd = clc.squared_mean_error(coord_pred, coord_true)
            print(f"[{protein['dataset']}] {protein['item']['prot_name']} ({i}) <{size}>  \t{rmsd} A*  \t{'OK' if rmsd < 10 else '_'}")
            return {'INDEX': i, 'SET': protein['dataset'], 'NAME': protein['item']['prot_name'], 'SIZE': size, 'RMSD': rmsd, 'CLL': cl, 'QVAR': qvar['var'], 'CVVAR': pred_cv['dev'] ** 2,
                    'QR1E': err_q_l1, 'QR2E': err_q_l2, 'QME': norm_err, 'CVR1E': err_cv_l1, 'CVR2E': err_cv_l2,
                    'V1AE': v1_angl, 'CVA0E': cv0_angl, 'CVA1E': cv1_angl, 'CVA': ref_cv['angle'], 'error': 0}
        except:
            print(f"[{protein['dataset']}] {protein['item']['prot_name']} ({i}) <{size}>")
            return {'INDEX': i, 'SET': protein['dataset'], 'NAME': protein['item']['prot_name'], 'SIZE': size, 'RMSD': np.NaN, 'CLL': np.NaN, 'QVAR': np.NaN, 'CVVAR': np.NaN,
                    'QR1E': np.NaN, 'QR2E': np.NaN, 'QME': np.NaN, 'CVR1E': np.NaN, 'CVR2E': np.NaN, 'V1AE': np.NaN,
                    'CVA0E': np.NaN, 'CVA1E': np.NaN, 'CVA': np.NaN, 'error': 1}


    #Цикл предсказания для CosMaP FCN5
    def predict_cosmap_fcn5(self):
        print('================\nPrediction start')
        pipe = self.load_pipe().build().cuda()
        data_name = self.config['inspect']['data']
        csv_name = f"predict_fcn5_{data_name}.csv"
        csv_path = os.path.join(self.config['attempt']['saves_dir'], csv_name)
        l = 0
        if data_name == 'train' or data_name == 'both':
            l += len(pipe.trn_dataset)
        if data_name == 'valid' or data_name == 'both':
            l += len(pipe.val_dataset)
        frame = pd.DataFrame(index = range(l), columns = ['INDEX', 'SET', 'NAME', 'SIZE', 'RMSD', 'CLL', 'QVAR', 'CVVAR',
                    'QR1E', 'QR2E', 'QME', 'CVR1E', 'CVR2E', 'V1AE', 'CVA0E', 'CVA1E',
                    'CVA', 'error'])
        i = 0
        if data_name == 'train' or data_name == 'both':
            while i < len(pipe.trn_dataset):
                #kostil
                try:
                    protein = {'dataset': 'train', 'index': i,
                               'item': pipe.trn_dataset._justload(i),
                               'input': pipe.trn_dataset._getitem_full(i)}
                    frame.loc[i] = self.predict_cosmap_fcn5_iteration(protein, i, pipe)
                except:
                    print('ERROR')
                    frame.loc[i]['error'] = 1
                i += 1
        i0 = i
        if data_name == 'valid' or data_name == 'both':
            while i < l:
                protein = {'dataset': 'valid', 'index': i - i0,
                           'item': pipe.val_dataset._justload(i - i0),
                           'input': pipe.val_dataset._getitem_full(i - i0)}
                frame.loc[i] = self.predict_cosmap_fcn5_iteration(protein, i, pipe)
                i += 1
        frame.to_csv(csv_path)
        print('================\nPrediction end')

    ################################################################################################
    #Итерация предсказания для CosMaP FCN5
    def predict_cosmap_fcn5_new_iteration(self, protein, i, pipeline):
        size = protein['item']['crop']
        homo = protein['item']['homo']
        name = protein['item']['id']
        #timer = time.time()
        try:
            with torch.no_grad():
                cosma = torch.tensor(np.expand_dims(protein['input']['inp']['cos_ma'], axis = (0,1))).cuda()
                amino1 = torch.tensor(np.expand_dims(protein['input']['inp']['amino1'], axis = 0)).cuda()
                amino2 = torch.tensor(np.expand_dims(protein['input']['inp']['amino2'], axis = 0)).cuda()
                result = pipeline(cosma, amino1, amino2).detach().cpu().numpy()[0]
            #
            cl = clc.cl_loss(0.8, 0.2, result, np.array([protein['input']['out']['cos_ma_pw'],
                                                         protein['input']['out']['cv'],
                                                         protein['input']['out']['cv_r']]))['loss']
            prot1, prot2 = protein['item']['coords']
            if homo:
                pred_stat = clc.full_predict_homo(prot1, result[0], result[1], result[2])
                true_stat = clc.full_predict_homo(prot1, protein['input']['out']['cos_ma_pw'],
                                                         protein['input']['out']['cv'],
                                                         protein['input']['out']['cv_r'])
            else:
                prot2 = protein['item']['align']['s'] @ prot2
                pred_stat = clc.full_predict_het(prot1, prot2, result[0], result[1], result[2])
                true_stat = clc.full_predict_het(prot1, prot2, protein['input']['out']['cos_ma_pw'],
                                                               protein['input']['out']['cv'],
                                                               protein['input']['out']['cv_r'])
            rmsd = clc.squared_mean_error(pred_stat['coords'][1], protein['item']['coords'][1])
            print(f"[{protein['dataset']}] {name} ({i}) <{size}> \t{rmsd} A*  \t[{cl}] \t{'OK' if rmsd < 10 else '_'} ({'HOM' if homo else 'het'})")
            return {'INDEX': i, 'SET': protein['dataset'], 'NAME': name, 'HOMO': homo, 'SIZE': size, 'RMSD': rmsd, 'CLL': cl,
                    'QVAR': pred_stat['q_dev'] ** 2, 'CVVAR': pred_stat['cv_dev'] ** 2, 'V1AE': clc.degree(pred_stat['q_v1'], true_stat['q_v1'], True),
                    'CVAE': clc.degree(pred_stat['cv_3d'], true_stat['cv_3d']), 'error': 0}
        except:
            print(f"[{protein['dataset']}] {name} ({i}) <{size}> {'HOM' if homo else 'het'}")
            return {'INDEX': i, 'SET': protein['dataset'], 'NAME': name, 'HOMO': homo, 'SIZE': size, 'RMSD': np.NaN, 'CLL': np.NaN, 'QVAR': np.NaN, 'CVVAR': np.NaN, 'V1AE': np.NaN, 'CVAE': np.NaN, 'error': 1}

    #Цикл предсказания для CosMaP FCN5 - NEW DATASET
    def predict_cosmap_fcn5_new(self):
        print('================\nPrediction start')
        pipe = self.load_pipe().build().cuda()
        data_name = self.config['inspect']['data']
        csv_name = f"predict_fcn5_new_{data_name}.csv"
        csv_path = os.path.join(self.config['attempt']['saves_dir'], csv_name)
        l = 0
        if data_name == 'train' or data_name == 'both':
            l += len(pipe.trn_dataset)
        if data_name == 'valid' or data_name == 'both':
            l += len(pipe.val_dataset)
        frame = pd.DataFrame(index = range(l), columns = ['INDEX', 'SET', 'NAME', 'HOMO', 'SIZE', 'RMSD', 'CLL', 'QVAR', 'CVVAR', 'V1AE', 'CVAE', 'error'])
        i = 0
        if data_name == 'train' or data_name == 'both':
            while i < len(pipe.trn_dataset):
                #kostil
                try:
                    protein = {'dataset': 'train', 'index': i,
                               'item': pipe.trn_dataset._justload(i),
                               'input': pipe.trn_dataset._getitem_full(i)}
                    frame.loc[i] = self.predict_cosmap_fcn5_new_iteration(protein, i, pipe)
                except:
                    print('ERROR')
                    frame.loc[i]['error'] = 1
                i += 1
        i0 = i
        if data_name == 'valid' or data_name == 'both':
            while i < l:
                protein = {'dataset': 'valid', 'index': i - i0,
                           'item': pipe.val_dataset._justload(i - i0),
                           'input': pipe.val_dataset._getitem_full(i - i0)}
                frame.loc[i] = self.predict_cosmap_fcn5_new_iteration(protein, i, pipe)
                i += 1
        frame.to_csv(csv_path)
        print('================\nPrediction end')

    ################################################################################################################

    #Прогон
    def progon(self, protein, pipeline):
        with torch.no_grad():
            result = pipeline(torch.tensor(np.expand_dims(protein['input']['inp'], axis = 0)).cuda()).detach().cpu().numpy()[0]
        n = result.shape[0]
        result = np.corrcoef(protein['item']['cos_ma'][0], result, rowvar = False)[0:n, n:(2*n)]
        return result

    #Экспериментируем эксперименты
    def experiment(self):
        print('<DEMO>')
        # print('================\nExperiment start')
        # pipeline = self.load_pipe()
        # pipeline.build().cuda()
        # #
        # #net = CosMaP_UNet(ver = self.config['unet']['version']).float()
        # #
        # for i in range(self.config['inspect']['count']):
        #     try:
        #         self.experiment_visualize(self.get_item(pipeline), i, pipeline) #net)
        #     except:
        #         print('ERROR [EXPERIMENT]')
        # print('================\nExperiment end')

    #Полное предсказание + сохранение файла

    def predict_and_save_iteration(self, data):
        print(f"[{self.index}] {data['id']} <{data['crop']}> [P&S]\n;;;;;;;;;;;;;;;;;")
        cosma = np.transpose(data['p_mtrxs']) @ data['p_mtrxs']
        res_idx = np.array([ndata.map_res2idx[x if x in ndata.all_res else 'UNK'] for x in data['resids']])
        amino = np.transpose(np.eye(len(ndata.all_res))[res_idx])
        with torch.no_grad():
            cosma_t = torch.tensor(np.expand_dims(cosma.astype(dtype=np.float32), axis = (0,1))).cuda()
            amino_t = torch.tensor(np.expand_dims(amino.astype(dtype=np.float32), axis = 0)).cuda()
            result = self.pipeline(cosma_t, amino_t, amino_t).detach().cpu().numpy()[0]
        tol = self.config['bumping']['tolerance']
        dist = self.config['bumping']['distance']
        prediction = clc.full_predict_homo(data['coords'], result[0], result[1], result[2], tol, dist)
        output = f"ID: \t\t{data['id']}\nLENGTH: \t{data['crop']}\nQ VARIATION: \t{prediction['q_dev']**2}\nCV VARIATION: \t{prediction['cv_dev']**2}\nPREFERED: \t{'Q' if prediction['prefer_q'] else 'CV'}\nTOLERANCE: \t{tol}"
        print(output)
        print('\n')
        return (clc.get_center(data['coords']), prediction['_q_mtrx'], prediction['_shift'], output)

    def predict_and_save(self):
        # print('================\nPrediction&Saving start')
        # if self.config['data']['new']:
        #     print('Unsupported yet [P&S]')
        # else:
        #     self.pipeline = self.load_pipe().cuda()
        #     self.index = 0
        #     pkls_file = self.config['data']['train'] if self.config['inspect']['data'] == 'train' else self.config['data']['valid']
        #     pkls = pd.read_csv(pkls_file, sep=';')
        #     pkls = pkls[pkls['crop'] >= self.config['hyperparams']['crop']]
        #     print(f"Loaded: {len(pkls)}")
        #     for pkl_path in random.choices(list(pkls['path']), k = self.config['inspect']['count']):
        #         try:
        #             id = os.path.basename(pkl_path).split('.')[0]
        #             path = f"{self.config['preproc']['pdb_dir']}{id}_raw.pdb"
        #             io.proceed_homo(id, path, self.predict_and_save_iteration, 'pdb_out/')
        #         except:
        #             print('ERROR [P&S]')
        #         self.index += 1
        # print('================\nPrediction&Saving end')
        print('<DEMO>')

    #Обработчик конфигурационного файла
    #[основной метод]
    def do(self):
        print('Start proccessing')
        if self.config['mode'] == 'preproc':
            result = self.preproc()
            self.config['data']['train'] = result['train_path']
            self.config['data']['valid'] = result['valid_path']
        elif self.config['mode'] == 'train':
            result = self.train()
            self.config['attempt']['checkpoint'] = result['new_checkpoint']
            self.config['attempt']['attempt'] = result['new_attempt']
            self.config['attempt']['epoch'] = result['new_epoch']
        elif self.config['mode'] == 'inspect':
            self.inspect()
            print('End proccessing')
            return
        # elif self.config['mode'] == 'duo':
        #     self.duo()
        #     print('End proccessing')
        #     return
        # elif self.config['mode'] == 'duo2':
        #     self.duo(2)
        #     print('End proccessing')
        #     return
        elif self.config['mode'] == 'predict':
            if self.config['net'] == 'cosmap_fcn5':
                if self.config['data']['new']:
                    self.predict_cosmap_fcn5_new()
                else:
                    self.predict_cosmap_fcn5()
            elif self.config['net'] == 'cosmap_unet':
                self.predict_cosmap_unet()
            elif self.config['net'] == 'dhd':
                print('DHD not maintained.')   
            else:
                self.predict(2 if self.config['net'] == 'cosmap' else 1)
            print('End proccessing')
            return
        elif self.config['mode'] == 'experiment':
            self.experiment()
            print('End proccessing')
            return
        elif self.config['mode'] == 'p&s':
            self.predict_and_save()
            print('End proccessing')
            return
        else:
            raise RuntimeError(f"Unknown mode '{self.config['mode']}' [LAUNCHER]")
        with open(self.config_path, 'w') as config_file:
            json.dump(self.config, config_file, indent = 4)
        print('End proccessing')


if __name__ == '__main__':
    config_path = 'new_cfg.json'
    launcher = CosMaP(config_path)
    launcher.do()