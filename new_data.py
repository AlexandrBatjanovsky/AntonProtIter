import os
import numpy as np
import pandas as pd
import pickle as pkl
import calculus as clc
from torch.utils.data import Dataset
import logging

model_type = np.float32 #and here

all_res = ('UNK', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
           'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU',
           'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
           'TRP', 'TYR', 'VAL')

#Код Анны для получения тензора остатков на лету

map_res2idx = {x: xi for xi, x in enumerate(all_res)}
map_idx2res = {y: x for x, y in map_res2idx.items()}

def get_pairwise_res_1hot_matrix(res: np.ndarray, res2idx: dict = None) -> np.ndarray:
    if res2idx is None:
        res2idx = map_res2idx
    res_idx = np.array([res2idx[x] for x in res])
    num_res = len(res2idx)
    X, Y = np.meshgrid(res_idx, res_idx)
    shp_inp = X.shape
    X = np.eye(num_res)[X.reshape(-1)]
    Y = np.eye(num_res)[Y.reshape(-1)]
    X = X.reshape(shp_inp + (num_res,))
    Y = Y.reshape(shp_inp + (num_res,))
    XY = np.dstack([X, Y])
    return XY

#Разные версии cosmap - всё равно cosmap
def is_cosmap(net):
    return net == 'cosmap' or net == 'cosmap01' or net == 'cosmap_mod' or net == "cosmap_mod2"

#Метод взятия 3х-мерного центрального вектора
def get_center_vec_3d(sample):
    center_vec = sample['coords'][1] - sample['coords'][0] 
    center_vec = np.mean(center_vec, axis = 0)
    return clc.normalize(center_vec)

#Создаёт input/output для cosmap_unet
def construct_cosmap_unet(sample, crop = None, calc_t = False, add_cv_r = False):
    #INP: CosMa
    cosma = sample['cos_ma'][0].astype(model_type)
    n = cosma.shape[0]

    #INP: Amino
    resid = sample['resid'][0]
    res_idx = np.array([map_res2idx[x] for x in resid])
    amino1 = amino2 = np.transpose(np.eye(len(all_res))[res_idx]).astype(model_type)

    #OUT: CosMa_pw
    cosmapw = sample['cos_ma_pw'][0].astype(model_type)

    #OUT: Central Vector
    chain = clc.normalize(clc.diff(clc.expand(np.transpose(sample['coords'][0]))))
    center_vec = get_center_vec_3d(sample)
    center_vec = np.expand_dims(clc.normalize(center_vec), axis = 1)
    center_vec = np.transpose(chain) @ center_vec
    cv = np.repeat(center_vec, repeats = n, axis = 1).astype(model_type)
    #RIGHT CV
    if add_cv_r:
        cv_r = np.transpose(cv)

    #CROP
    if not crop is None:
        if not type(crop) is tuple:
            crop = (crop, crop)
        i = np.random.randint(0, n - crop[0]) if crop[0] < n else 0
        j = np.random.randint(0, n - crop[1]) if crop[1] < n else 0
        cosma = cosma[i:(i + crop[0]), j:(j + crop[1])]
        cosmapw = cosmapw[i:(i + crop[0]), j:(j + crop[1])]
        cv = cv[i:(i + crop[0]), j:(j + crop[1])]
        #RIGHT CV
        if add_cv_r:
            cv_r = cv_r[i:(i + crop[0]), j:(j + crop[1])]
        amino1 = amino1[:, j:(j + crop[1])] #i, j OK?     ---    YOS
        amino2 = amino2[:, i:(i + crop[0])]
        
        #T MATRICES
        if calc_t:
            lchain = chain[:, i:(i + crop[0])]
            rchain = chain[:, j:(j + crop[1])]
            t1 = np.transpose(lchain) @ np.linalg.inv(lchain @ np.transpose(lchain))
            t2 = np.transpose(rchain) @ np.linalg.inv(rchain @ np.transpose(rchain))

    elif calc_t:
        #T MATRICES
        t1 = t2 = np.transpose(chain) @ np.linalg.inv(chain @ np.transpose(chain))

    result =  {'inp': {'cos_ma': cosma, 'amino1': amino1, 'amino2': amino2},
               'out': {'cos_ma_pw': cosmapw, 'cv': cv}}
    if calc_t:
        result['inp']['t1'] = t1
        result['inp']['t2'] = t2
    if add_cv_r:
        result['out']['cv_r'] = cv_r
    return result

    

#Класс датсета для CosMaP и DeepHD
class CosMaPDataset(Dataset):
    def __init__(self, path_idx, crop_size, net, channels = [], calc_t = False, add_cv_r = False):
        super().__init__()
        self.path_idx = path_idx
        self.crop_size = crop_size
        if not is_cosmap(net) and net != 'dhd' and net != 'cosmacvp' and net != 'cosmaprime' and net != 'unet' and net != 'cosmap_unet' and net != 'cosmap_fcn5':
            raise RuntimeError(f"'{net}' unknown net type [DATA]")
        self.net = net
        self.train_e = False
        self.cosmap_unet = net == 'cosmap_unet'
        self.cosmap_fcn5 = net == 'cosmap_fcn5'
        if not self.cosmap_unet and not self.cosmap_fcn5:
            self.use_dst = 'dst' in channels
            self.use_cos_ma = 'cos_ma' in channels
            self.use_resid = 'resid' in channels
            self.use_cos_ma_pw = 'cos_ma_pw' in channels
            self.calc_t = calc_t
            if not (self.use_dst or self.use_cos_ma or self.use_resid):
                raise RuntimeError("no 'dst', 'cos_ma(_pw)' or 'resid' in channels [DATA]")
        self.data = None
        self.calc_t = calc_t
        self.add_cv_r = add_cv_r

    #build полностью переезжает из DHDDataset (как и почти всё здесь присутствующее)
    def build(self):
        self.wdir = os.path.dirname(self.path_idx) #папка, где лежит csv
        self.data_idx = pd.read_csv(self.path_idx, sep=';') #csv с названиями pkl файлов (я поставил сепаратор ;)
        self.data_idx['path_abs'] = [os.path.join(self.wdir, x) for x in self.data_idx['path']] #добавляет абсолютные пути
        logging.info('\t::load paths: #samples = {}'.format(len(self.data_idx)))
        self.data = self.data_idx[self.data_idx['crop'] >= self.crop_size].sample(frac=1, axis=0, ignore_index=True) #frac!
        logging.info('\t\t\tloaded: #samples={} with size >= {}'
                     .format(len(self.data), self.crop_size))
        logging.info('self.data.shape = {}'.format(self.data.shape))
        return self

    def __len__(self):
        return len(self.data.index)

    #Берёт из матриц рандомную подматрицу нужного размера
    def _get_random_crop(self, dst_info: dict, crop_size: int) -> dict:
        nrc = dst_info['inp'].shape[0]
        if crop_size < nrc:
            rr, rc = np.random.randint(0, nrc - crop_size, 2)
            inp_crop = dst_info['inp'][rr: rr + crop_size, rc: rc + crop_size, ...]
            out_crop = dst_info['out'][rr: rr + crop_size, rc: rc + crop_size, ...]
        else:
            inp_crop = dst_info['inp']
            out_crop = dst_info['out']
        if self.net == 'unet':
            out_crop = out_crop.transpose((2, 0, 1))
        ret = {
            'inp': inp_crop.transpose((2, 0, 1)),
            'out': out_crop
        }
        return ret

    def _construct_inp_out(self, sample):
        #input
        inp = []
        if self.use_dst:
            if self.net != "cosmap_mod" and self.net != "cosmap_mod2":
                inp.append(sample['dst'][0])
            else:
                inp.append(np.log(sample['dst'][0] + 1))
        if self.use_cos_ma:
            inp.append(sample['cos_ma'][0])
        if self.use_resid:
            inp.append(get_pairwise_res_1hot_matrix(sample['resid'][0]).astype(model_type))
        if self.use_cos_ma_pw:
            inp.append(sample['cos_ma_pw'][0]) #для обучения сетки используем настоящие межмолекулярные матрицы косинусов
        inp = np.dstack(inp)
        #output
        if self.train_e or self.net == 'cosmaprime':
            out = sample['cos_ma'][0]
        elif is_cosmap(self.net):
            out = sample['cos_ma_pw'][0]
        elif self.net == 'dhd':
            out = sample['inter'][0].astype(model_type)
        elif self.net == 'cosmacvp' or self.net == 'unet':
            center_vec = get_center_vec_3d(sample)
            center_vec = np.expand_dims(clc.normalize(center_vec), axis = 1)
            center_vec = np.transpose(clc.normalize(clc.diff(clc.expand(np.transpose(sample['coords'][0]))))) @ center_vec
            out = np.repeat(center_vec, repeats = center_vec.shape[0], axis = 1) #ВЫДЕЛИТЬ В ОТДЕЛЬНЫЙ МЕТОД
            if self.net == 'unet':
                out = np.dstack([sample['cos_ma_pw'][0], out])
        return {'inp': inp, 'out': out}

    def _getitem_path(self, item):
        return self.data.loc[item, 'path']

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError()
        sample_path = self._getitem_path(item)
        try:
            sample = pkl.load(open(sample_path, 'rb'))
            #cosmap_unet
            if self.cosmap_unet:
                return construct_cosmap_unet(sample, self.crop_size, self.calc_t, self.add_cv_r)
            #cosmap_fcn5
            if self.cosmap_fcn5:
                return construct_cosmap_unet(sample, self.crop_size, False, True)
            #
            dst_info = self._construct_inp_out(sample)
            dst_info = self._get_random_crop(dst_info, crop_size=self.crop_size)
            dst_info['data_index'] = self.data.iloc[item, 0]
            logging.info('{}: inp-shape/out-shape = {}/{} ({})'.format(item, dst_info['inp'].shape, dst_info['out'].shape, dst_info['data_index']))
            return dst_info
        except Exception as exc:
            print(f'\n...!{exc}. item: {item}. path {sample_path}!')
            return self.__getitem__(item + 1)

    #взятие без кропа
    def _getitem_full(self, item):
        if item >= len(self):
            raise IndexError()
        sample_path = self._getitem_path(item)    
        try:
            sample = pkl.load(open(sample_path, 'rb'))
            #cosmap_unet
            if self.cosmap_unet:
                return construct_cosmap_unet(sample, calc_t = self.calc_t, add_cv_r = self.add_cv_r)
            #cosmap_fcn5
            if self.cosmap_fcn5:
                return construct_cosmap_unet(sample, calc_t = False, add_cv_r = True)
            #
            dst_info = self._construct_inp_out(sample)
            dst_info['inp'] = dst_info['inp'].transpose((2, 0, 1))
            if self.net == 'unet':
                dst_info['out'] = dst_info['out'].transpose((2, 0, 1))
            dst_info['data_index'] = self.data.iloc[item, 0]
            logging.info('{}: inp-shape/out-shape = {}/{} ({})'.format(item, dst_info['inp'].shape, dst_info['out'].shape, dst_info['data_index']))
            return dst_info
        except Exception as exc:
            print(f'\n...!{exc}. item: {item}. path {sample_path}!')
            return self.__getitem__(item + 1)

    def _justload(self, item):
        return pkl.load(open(self._getitem_path(item), 'rb'))

    def _get_channels_count(self):
        channel = 0
        if self.use_dst:
            channel += 1
        if self.use_cos_ma:
            channel += 1
        if self.use_resid:
            channel += 42
        if self.use_cos_ma_pw:
            channel += 1
        return channel

##################################################
##################################################

#Новый датасет - на новых началах
def read_new_pkl(path):
    with open(path, 'rb') as f:
        return pkl.load(f)

#inp-out
def new_inp_out(loaded, crop = None):
    #COORDS
    vecs0 = loaded['coords'][0]
    vecs1 = loaded['coords'][1]

    #SIZES
    n0 = len(vecs0[0])
    n1 = len(vecs1[0])

    #CROP
    if not crop is None:
        if not type(crop) is tuple:
            crop = (crop, crop)
        crop = (min(crop[0], n0), min(crop[1], n1))

        i0 = np.random.randint(0, n0 - crop[0]) if crop[0] < n0 else 0
        i1 = np.random.randint(0, n1 - crop[1]) if crop[1] < n1 else 0
    else:
        crop = (n0, n1)
        i0 = 0
        i1 = 0

    #RESIDUES
    res_idx0 = np.array([map_res2idx[x if x in all_res else 'UNK'] for x in loaded['resids'][0][i0:i0 + crop[0]]])
    res_idx1 = np.array([map_res2idx[x if x in all_res else 'UNK'] for x in loaded['resids'][1][i1:i1 + crop[1]]])
    amino0 = np.transpose(np.eye(len(all_res))[res_idx0])
    amino1 = np.transpose(np.eye(len(all_res))[res_idx1])


    #P MATRIXES
    p0 = loaded['p_mtrxs'][0][:, i0:i0 + crop[0]]  #swap 0 and 1?
    p1 = loaded['p_mtrxs'][1][:, i1:i1 + crop[1]]
    cosmapw = np.transpose(p0) @ p1

    #CV
    cv = clc.get_center_vector(vecs0, vecs1)
    left_cv = np.repeat(np.transpose(p0) @ cv, repeats=crop[1], axis=1)
    right_cv = np.repeat(np.transpose(-cv) @ p1, repeats=crop[0], axis=0)

    if loaded['homo']:
        cosma = np.transpose(p0) @ p0
    else:
        cosma = np.transpose(p0) @ (loaded['align']['s'] @ p1) #loaded['s_t'][0] @ p0) @ (loaded['s_t'][1] @ p1
       
    return {'inp': {'cos_ma': cosma.astype(model_type),
                    'amino2': amino0.astype(model_type), 'amino1': amino1.astype(model_type)},
            'out': {'cos_ma_pw': cosmapw.astype(model_type),
                    'cv': left_cv.astype(model_type),
                    'cv_r': right_cv.astype(model_type)}}


#Новый датасет
class CosMaP_FCN5_Dataset(Dataset): #DATASET ENTIRELY IN MEMORY!!!!
    def __init__(self, path_idx, crop_size, check_align, load_all = True):
        super().__init__()
        self.path_idx = path_idx
        self.crop_size = crop_size
        self.check_align = check_align
        self.data = None
        self.load_all = load_all
        self.loaded = None

    #COPYPASTE
    def build(self, print_size = True):
        self.wdir = os.path.dirname(self.path_idx) #папка, где лежит csv
        self.data_idx = pd.read_csv(self.path_idx) #csv с названиями pkl файлов
        #Используются абсолютные пути (path)
        self.data = self.data_idx[self.data_idx['crop'] >= self.crop_size]
        if self.check_align:
            #SLOW
            self.data = pd.DataFrame([r for i, r in self.data.iterrows() if r['homo'] or clc.check_goodness(r['sim'], r['sd'])])
        self.data = self.data.sample(frac=1, axis=0, ignore_index=True)
        if print_size:
            print(f"DATASET SIZE: {len(self.data)}")
        if self.load_all:
            self.loaded = [read_new_pkl(path) for path in self.data['path']]
            print('\tLOADED')
        return self

    #
    def __len__(self):
        return len(self.data.index)

    def _getitem_path(self, item): #load model 0 (пока так)
        return self.data.loc[item, 'path']

    def _get_channels_count(self):
        return 41
    #

    def _justload(self, item):
        return read_new_pkl(self._getitem_path(item)) if not self.load_all else self.loaded[item]

    def __getitem__(self, item):
        return new_inp_out(self._justload(item), self.crop_size)

    def _getitem_full(self, item):
        return new_inp_out(self._justload(item))

    #Откорректировать!!!