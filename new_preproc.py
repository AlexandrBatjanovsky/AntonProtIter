import os
import numpy as np
import pickle as pkl
import scipy.spatial.distance as sdst
from Bio.PDB import PDBParser


blacklist = ['3ean', '1kp0', '3u5s'] #Белки, вызывавшие проблемы в прошлом

#Код кастомизируемого препроцессинга
def read_prot(path_pdb, type = 'homodimer', pdb_parser = None, atoms_type = 'beta',
    calc_dst = True, calc_dst_pw = False, calc_cos_ma = True, calc_cos_ma_pw = True,
    calc_resid = True, calc_inter = True, inter_border = 8, calc_coords = True,
    cut_two = True, consider_blacklist = True, model_type = np.float32):
    if pdb_parser is None:
        pdb_parser = PDBParser(QUIET = True)

    #Считывание pdb файла
    prot_name = os.path.basename(path_pdb).replace('_raw', '').replace('.pdb', '')
    if consider_blacklist and prot_name in blacklist:
        raise RuntimeError(f'Blacklisted protein ({prot_name}) [PREPROC]')
    models_ = list(pdb_parser.get_structure(prot_name, path_pdb).get_models())
    models = []
    for m in models_:
        for c in list(m.get_chains()):
            models.append(c)
    if len(models) < 2 and type != 'mono':
        raise RuntimeError(f'Less then 2 models ({len(models)}) [PREPROC]')

    if type == 'mone':
        models = [models[0]]
    elif cut_two:
        models = models[0:2]

    #Читаем атомы
    if atoms_type == 'beta':
        atoms = [[x for x in m.get_atoms() if ((x.get_parent().resname == 'GLY' and x.name == 'CA') or
            (x.name == 'CB')) and (not x.get_full_id()[3][0].strip())] for m in models]
    elif atoms_type == 'alpha':
        atoms = [[x for x in m.get_atoms() if x.name == 'CA' and (not x.get_full_id()[3][0].strip())]
            for m in models]
    else:
        raise RuntimeError(f'Unknown chain type ({atoms_type}) [PREPROC]')

    #Имена аминокислот
    resid = [[x.get_parent().resname for x in a] for a in atoms] if type == 'homodimer' or calc_resid else None
    
    #Проверка на гомодимерность
    if type == 'homodimer':
        if not all(resid[0] == res for res in resid):
            raise RuntimeError(f"'{type}' conditions unfulfilled [PREPROC]")
    elif type != 'heterodimer' and type != 'mono':
        raise RuntimeError(f"'({type})' unknown type [PREPROC]")

    #Координаты
    coords = [np.array([x.coord for x in a]).astype(model_type) for a in atoms]

    #Матрицы расстояний
    dst = np.stack([sdst.cdist(x, x, 'euclidean') for x in coords]).astype(model_type) if calc_dst else None
    dst_pw = None
    if calc_dst_pw or calc_inter:
        dst_pw = []
        for i in range(0, len(coords)):
            for j in range(i + 1, len(coords) - i):
                dst_pw.append(sdst.cdist(coords[i], coords[j], 'euclidean').astype(model_type))
    
    #Матрицы косинусов
    cos_ma = None
    cos_ma_pw = None
    if calc_cos_ma or calc_cos_ma_pw:
        diffs = [np.diff(np.row_stack((x, x[0, :])), axis = 0).astype(model_type) for x in coords]
        if calc_cos_ma:
            cos_ma = 1 - np.stack([sdst.cdist(x, x, 'cosine').astype(model_type) for x in diffs]) #rewrite!!!!!
        if calc_cos_ma_pw:
            cos_ma_pw = []
            for i in range(0, len(diffs)):
                for j in range(i + 1, len(diffs) - i):
                    cos_ma_pw.append(1 - sdst.cdist(diffs[i], diffs[j], 'cosine').astype(model_type)) #np.corrcoef!!!

    #Интерфейсы
    inter = [m < inter_border for m in dst_pw] if calc_inter else None
    
    return {'prot_name': prot_name, 'type': type, 'path_pdb': path_pdb, 'atoms_type': atoms_type,
            'cut_two': cut_two, 'inter_border': inter_border, 'model_type': model_type,
            'dst': dst if calc_dst else None,
            'dst_pw': dst_pw if calc_dst_pw else None,
            'cos_ma': cos_ma if calc_cos_ma else None,
            'cos_ma_pw': cos_ma_pw if calc_cos_ma_pw else None,
            'resid': resid if calc_resid else None,
            'inter': inter if calc_inter else None,
            'coords': coords if calc_coords else None}


#Конверитрует pdb файл в сериализованный dict 
#index и dataset_length - для вывода в консоль
def read_and_dump(path_pdb, out_dir, index = 0, dataset_length = 0, type = 'homodimer',
    pdb_parser = None, atoms_type = 'beta',
    calc_dst = True, calc_dst_pw = False, calc_cos_ma = True, calc_cos_ma_pw = True,
    calc_resid = True, calc_inter = True, inter_border = 8, calc_coords = True,
    cut_two = True, consider_blacklist = True, model_type = np.float32):
    try:
        prot = read_prot(path_pdb, type, pdb_parser, atoms_type, calc_dst, calc_dst_pw,
                         calc_cos_ma, calc_cos_ma_pw, calc_resid, calc_inter, inter_border,
                         calc_coords, cut_two, consider_blacklist, model_type)
        print(f'[{index}/{dataset_length}]\n', end = '')
    except Exception as err:
        print(f'<{err}>\n', end = '')
        return None
    
    path_out = os.path.join(out_dir, prot['prot_name'] + '.pkl')
    prot_crop = len(prot['resid'][0])
    with open(path_out, 'wb') as f:
        pkl.dump(prot, f)
    return [path_out, prot_crop]