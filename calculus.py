import numpy as np
import numpy.linalg as lg
import scipy.spatial.distance as sdst

#Блок с необходимой линейной алгеброй

#vecs - 3 x n

def length(vecs):
    return np.expand_dims(lg.norm(vecs, ord = 2, axis = 0), axis = 0)

def diff(vecs):
    return np.diff(vecs, axis = 1)

def normalize(vecs):
    return vecs / np.repeat(length(vecs), repeats = 3, axis = 0)

def expand(vecs):
    return np.column_stack([vecs, vecs[:, 0]])

def get_n(vecs):
    return vecs.shape[1]

def get_center(vecs):
    return np.expand_dims(np.mean(vecs, axis = 1), axis = 1)

def centralize(vecs):
    return vecs - np.repeat(get_center(vecs), repeats = get_n(vecs), axis = 1)

def get_revs(vecs):
    return [lg.inv(vecs[:, j:(j + 3)]) for j in range(0, get_n(vecs) - 2)] #не numpy массив!!!

def get_q_tens_reduced(revs, cos_ma_pw): #Not working smh
    trevs0 = np.transpose(revs[0]) #reshaping
    return np.expand_dims(np.array([trevs0 @ cos_ma_pw[0:3, j:(j + 3)] @ revs[j]
                          for j in range(0, len(revs))]), axis = 0)

def get_q_tens(revs, cos_ma_pw):
    return np.array([[np.transpose(revs[i]) @ cos_ma_pw[i:(i + 3), j:(j + 3)] @ revs[j]
           for j in range(0, len(revs))] for i in range(0, len(revs))])

def mtrx_sd(mtrxs, mean, axis = (1, 2)):
    expanded = np.expand_dims(mean, axis = 0)
    reps = np.repeat(expanded, mtrxs.shape[0], axis = 0)
    norms = lg.norm(mtrxs - reps, ord = 2, axis = axis)
    return np.sqrt(np.mean(norms ** 2))
    #return np.sqrt(np.mean(lg.norm(mtrxs - np.repeat(np.expand_dims(mean, axis = 0), mtrxs.shape[0], axis = 0), axis = 0)))

def mtrx_mean(mtrxs):
    return np.mean(mtrxs, axis = 0)

def mtrx_conds(mtrxs, bord = 10):
    return [lg.cond(mtrxs[k, :, :]) < bord for k in range(0, mtrxs.shape[0])]

#############################################################################################
#############################################################################################
# LEGACY - пусть будет

#vecs - координаты атомов в правильной матрице
def get_q(vecs, cos_ma_pw, reduced = False, bord = 10):
    revs = get_revs(normalize(diff(expand(vecs))))
    tens = None
    if False:#reduced:
        tens = get_q_tens_reduced(revs, cos_ma_pw)
    else:
        tens = get_q_tens(revs, cos_ma_pw)
    means = np.array([mtrx_mean(tens[:, j, :, :]) for j in range(0, tens.shape[1])])
    consists = np.array([mtrx_sd(tens[:, j, :, :], means[j, :, :]) for j in range(0, tens.shape[1])])
    conds = mtrx_conds(means, bord) #порождает ошибки!
    cond_stat = 1 - np.mean(conds) #outcast rate
    #
    if cond_stat == 1:
        print(f"### OUTCAST RATE = 1 ###") 
    #
    mean = mtrx_mean(means[conds, :, :])
    consist = np.mean(consists[conds])
    deviat = mtrx_sd(means[conds, :, :], mean)
    return {'Q': mean, 'cons': consist, 'dev': deviat, 'cond': cond_stat}

#Вырвем пожалуй...
def get_vec1(mtrx):
    w, v = lg.eig(mtrx)
    ind = np.argmin(np.abs(1 - w))
    return np.real(v[:, ind])

#def get_vec1_2(mtrx):
#    return get_vec1((mtrx + np.transpose(mtrx)) / 2)

def refine_q(q):
    vec1 = np.expand_dims(get_vec1(q), axis = 1)
    s_inv = lg.qr(vec1, mode = 'complete')[0]
    s = np.transpose(s_inv)
    q_ref = s_inv @ np.diag([1, -1, -1]) @ s
    return {'Q_ref': q_ref, 'S': s, 'vec1': vec1}

#
#Упрощение...
#

def easy_q(chain, cos_ma):
    return refine_q(get_q(chain, cos_ma)['Q'])['Q_ref']

def get_matrices(vecs, cos_ma_pw, amount = -1, cond_border = 10):
    revs = get_revs(normalize(diff(expand(vecs))))
    tens = get_q_tens(revs, cos_ma_pw)
    tens = tens.reshape((tens.shape[0] * tens.shape[1], 3, 3))
    #condition filter
    tens = tens[np.array([lg.cond(tens[i]) < cond_border for i in range(tens.shape[0])])]
    #may add more filters
    if amount != -1:
        tens = tens[np.random.choice(tens.shape[0], amount)] #bit of optimization
    return tens

def get_vectors(vecs, cos_ma_pw, amount = -1, bord = 10):
    mtrxs = get_matrices(vecs, cos_ma_pw, amount, bord)
    vecs1 = np.array([get_vec1(mtrxs[i]) for i in range(mtrxs.shape[0])])
    return vecs1

#
#Унификация межмолекулярной матрицы косинусов
#

def unificate(p1, q): #Весьма бесполезно
    return np.transpose(p1) @ q @ p1 #можно было и не выделять...

#
#Вычисление центральных векторов для CosMa(cv)P
#

def get_cv_tens(revs, cv_image):
    return np.array([[normalize(np.transpose(revs[i]) @ cv_image[i:i+3, j])
        for j in range(len(revs))] for i in range(len(revs))])

def get_cvs(vecs, cv_image, amount = -1, left_border = 0.5, right_border = 1): #should work TO DO
    revs = get_revs(normalize(diff(expand(vecs))))
    tens = get_cv_tens(revs, cv_image)
    tens = tens.reshape((tens.shape[0] * tens.shape[1], 3))
    #tens = np.transpose(normalize(np.transpose(tens)))
    if amount != -1:
        tens = tens[np.random.choice(tens.shape[0], amount)]
    return tens

def get_cv(vecs, cv_image): #Возможно не самый оптимальный способ... (Ориентация по столбцам/строкам!)
    revs = get_revs(normalize(diff(expand(vecs))))
    tens = get_cv_tens(revs, cv_image)
    #stats
    means = np.array([normalize(mtrx_mean(tens[:, j, :])) for j in range(0, tens.shape[1])])
    consists = np.array([mtrx_sd(tens[:, j, :], means[j, :], (1)) for j in range(0, tens.shape[1])])
    mean = normalize(mtrx_mean(means))
    consist = np.mean(consists)
    deviat = mtrx_sd(means, mean, (1))
    return {'center_vec': mean, 'cons': consist, 'dev': deviat}

def get_cv2(vecs, cv_image): #может переводить в полярные координаты?
    revs = get_revs(normalize(diff(expand(vecs))))
    tens = get_cv_tens(revs, cv_image)
    #
    tens = np.reshape(tens, (tens.shape[0] * tens.shape[1], 3, 1))
    mean = normalize(mtrx_mean(tens))
    deviat = mtrx_sd(tens, mean, (1))
    return {'center_vec': mean, 'dev': deviat}

#
#Ортогонализация пары Q_vec/CV_vec
#

#Ортагонализация с опорой на единичную ось Q
def ortagonalize(q_vec, cv_vec):
    q_cv_cos = (np.transpose(q_vec) @ cv_vec)[0, 0]
    refined = normalize(cv_vec - q_vec * q_cv_cos)
    return {'center_vec': refined, 'angle': np.arccos(np.abs(q_cv_cos)) * 180 / np.pi}

#########################################################################################################
#########################################################################################################


#
#Составление полной модели белка
#

def rotate_prot(vecs, q):
    p1 = diff(vecs)
    p2 = q @ p1
    vecs2 = np.column_stack([np.zeros(shape = (3, 1)), p2])
    vecs2 = np.cumsum(vecs2, axis = 1)
    vecs2 = centralize(vecs2) + np.repeat(get_center(vecs), repeats = get_n(vecs2), axis = 1)
    return vecs2

def shift_prot(vecs, cv, dist):
    vecs2 = vecs + np.repeat(dist * cv, repeats = get_n(vecs), axis = 1)
    return vecs2

#
#Стукивание белков
#

def f(vecs1, vecs2, bord):
    dists = np.ravel(sdst.cdist(np.transpose(vecs1), np.transpose(vecs2), 'euclidean'))
    n = dists.size
    dists = dists[dists < bord]
    if dists.size == 0:
        return 0
    return np.sum(1 / dists ** 2 - np.repeat(1 / bord ** 2, len(dists))) / n

def calc_f(vecs1, vecs2, bord, cv, start = 0.1, shift = 0.1, end = 50):
    m = (end - start) / shift + 1
    return np.array([f(vecs1, shift_prot(vecs2, cv, start + shift * i), bord) for i in range(round(m))])

#Первый ноль функции
def find_first0(vecs1, vecs2, bord, cv, zero_estim = 0.0, start = 0.1, shift = 0.1, end = 400):
    m = (end - start) / shift + 1
    i = 0
    while i < m:
        curr_shift = start + shift * i
        curr_vecs = shift_prot(vecs2, cv, curr_shift)
        #dists = np.ravel(sdst.cdist(np.transpose(vecs1), np.transpose(curr_vecs), 'euclidean'))
        #if np.all(dists >= bord):
        f_val = f(vecs1, curr_vecs, bord)
        if f_val <= zero_estim:
            return {'shift': curr_shift, 'f': f_val}
        else:
            i += 1
    return {'shift': end, 'f': -1}

#
#Ошибка модели
#

def squared_mean_error(vecs1, vecs2):
    return np.sqrt(np.mean(length(vecs1 - vecs2) ** 2))

#
#Матрички для тестов
#

def euler_angles(alph, beth, gamm):
    return np.array([[np.cos(alph), -np.sin(alph), 0],
                     [np.sin(alph),  np.cos(alph), 0],
                     [0,             0,            1]]) @ np.array(
                    [[1,             0,            0],
                     [0, np.cos(beth), -np.sin(beth)],
                     [0, np.sin(beth),  np.cos(beth)]]) @ np.array(
                    [[np.cos(gamm), 0, -np.sin(gamm)],
                     [0,            1,             0],
                     [np.sin(gamm), 0,  np.cos(gamm)]])

def gen_q(alph, beth):
    vec1 = np.array([np.cos(alph) * np.cos(beth),
                     np.cos(alph) * np.sin(beth),
                     np.sin(alph)])
    vec1 = np.expand_dims(vec1, axis = 1)
    s_inv = lg.qr(vec1, mode = 'complete')[0]
    return s_inv @ np.diag([1, -1, -1]) @ np.transpose(s_inv)

def rand_gen_s():
    return euler_angles(np.random.rand() * 2 * np.pi, np.random.rand() * 2 * np.pi, np.random.rand() * 2 * np.pi)

def rand_gen_q():
    return gen_q(np.random.rand() * 2 * np.pi, (np.random.rand() - 0.5) * np.pi)


#############
#FAST METHOD#
#############

def fast_q(vecs, cos_ma_pw):
    chain = normalize(diff(expand(vecs)))
    matrx = lg.inv(chain @ np.transpose(chain))
    q = matrx @ chain @ cos_ma_pw @ np.transpose(chain) @ matrx
    q = (q + np.transpose(q)) / 2 #!
    w, v = lg.eigh(q)
    v_inv = np.transpose(v)
    q_ref = v @ np.diag([-1, -1, 1]) @ v_inv
    return {'Q': q, 'Q_ref': q_ref, 'S': v_inv, 'vec1': np.expand_dims(v[:, 2], axis = 1)}

#no use
def fast_a(chain, cos_ma_pw):
    matrx = lg.inv(chain @ np.transpose(chain))
    return matrx @ chain @ cos_ma_pw @ np.transpose(chain) @ matrx

### wrong ###
def fast_q_het(chain1, chain2, cosma_ch1ch2):
    t1 = lg.inv(chain1 @ np.transpose(chain1)) @ chain1
    t2 = np.transpose(chain2) @ lg.inv(chain2 @ np.transpose(chain2))
    q = t1 @ cosma_ch1ch2 @ t2
    q = (q + np.transpose(q)) / 2
    u, v = lg.eigh(q)
    if np.prod(u) < 0:
        least = np.argmin(np.abs(u))
        u[least] = -u[least]
    v_inv = np.transpose(v)
    u = np.sign(u)
    q = v @ np.diag(u) @ v_inv
    return {'Q': q, 'S': v_inv, 'vec1': np.expand_dims(v[:, 2], axis = 1),
            'cosma': np.transpose(chain1) @ q @ chain2, 'eigs': u}

def left_cv(vecs, cv_mtrx):
    chain = normalize(diff(expand(vecs)))
    cvs = lg.inv(chain @ np.transpose(chain)) @ chain @ cv_mtrx #не оптимально, быстрее - сначала среднее, потом умножение
    cvs = normalize(cvs)
    cv = normalize(np.mean(cvs, axis = 1))
    cv = np.expand_dims(cv, axis = 1)
    deviat = mtrx_sd(cvs, cv, (1))
    return {'center_vec': cv, 'dev': deviat}

def fast_a2(chain1, chain2, cos_ma_pw): #p1_a_p2 !!! kinda bad      ------------       сплошная неразбериха...
    matrx1 = lg.inv(chain1 @ np.transpose(chain1))
    matrx2 = lg.inv(chain2 @ np.transpose(chain2))
    return matrx1 @ chain1 @ cos_ma_pw @ np.transpose(chain2) @ matrx2

def right_cv(vecs, cv_r_mtrx):
    return left_cv(vecs, np.transpose(cv_r_mtrx))


#############################################################
def svd_a(p1, p2, cosma, only_positive = True):                 # BRAND NEW
    t1 = lg.inv(p1 @ np.transpose(p1)) @ p1
    t2 = np.transpose(p2) @ lg.inv(p2 @ np.transpose(p2))
    a = t1 @ cosma @ t2
    u, diag, vh = lg.svd(a)
    if only_positive and lg.det(a) < 0:
        least = np.argmin(np.abs(diag))
        diag[least] = -diag[least]
    return u @ np.diag(np.sign(diag)) @ vh
#############################################################

#########
#CL Loss#
#########

#power!!!
def cl_loss(alph, beth, x, y):
    y2 = np.sum(y ** 2)
    x2 = np.sum(x ** 2)
    xy = np.sum(x * y)
    C = (y2 - xy) / y2
    L = (y2 - x2) / y2
    return {'loss': alph * C ** 2 + beth * np.abs(L),
            'C': C, 'L': L}    


##############
#HETERODIMERS#      UNUSED
##############

#Диагонали матриц
def get_diagonals(mtrx):
     m, n = mtrx.shape
     return [np.diag(mtrx, k) for k in range(-m + 1, n)]

#Статистика подобия (наивная)
def cosma_goodness(cosma, border = 0.3):
    m, n = cosma.shape
    sum = 0
    for diag in get_diagonals(cosma):
        if np.mean(diag) > border:
            sum += len(diag)
    return sum / (m + n - 1)

#Натуральная система координат
def natural_coords(vecs):
    chain = normalize(diff(expand(vecs)))
    ppt = chain @ np.transpose(chain)
    u, v = lg.eigh(ppt)
    if lg.det(v) < 0:
        v[:, 0] = -v[:, 0]
    p_nat = np.transpose(v) @ chain
    return {'p_nat': p_nat, 's_t': v, 'eig': u}

#Правильно соориентированные цепи 
def pair_orient(vecs1, vecs2):
    prot1 = natural_coords(vecs1)
    prot2 = natural_coords(vecs2)
    goodness = []
    cosmas = []
    prots2 = []
    rots = []
    for rot in [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]:
        rot = np.diag(rot)
        prot2rot = rot @ prot2['p_nat']
        prots2.append(prot2rot)
        rots.append(prot2['s_t'] @ rot)
        cosma = np.transpose(prot2rot) @ prot1['p_nat']
        cosmas.append(cosma)
        goodness.append(cosma_goodness(cosma))
    idx = np.argmax(goodness)
    return {'vecs':   [vecs1, vecs2],
            'chains': [prot1['p_nat'], prots2[idx]],
            's_t':    [prot1['s_t'],   rots[idx]],
            'eigs':   [prot1['eig'],   prot2['eig']],
            'cosma': cosmas[idx], 'goodness': goodness[idx]}

############
#Additional#
############

#Метод для получения cosma из vecs
def get_cosma(vecs1, vecs2 = None):
    chain1 = normalize(diff(expand(vecs1)))
    if vecs2 is None:
        chain2 = chain1
    else:
        chain2 = normalize(diff(expand(vecs2)))
    return np.transpose(chain1) @ chain2 #P2^t @ P1 NOT!!!!

def get_center_vector(vecs1, vecs2):
    c1 = np.mean(vecs1, 1, keepdims=True)
    c2 = np.mean(vecs2, 1, keepdims=True)
    return normalize(c2 - c1)

###################3

#Blur measurment!
def cosma_var(vecs, q, cosma):
    chain = normalize(diff(expand(vecs)))
    cosma_est = np.transpose(chain) @ q @ chain
    err = cosma - cosma_est #не модуль - чтобы можно было на нормальность проверять
    return {'var': np.mean(err ** 2), 'abs': np.mean(np.abs(err)),
            'cosma': cosma_est, 'err': err}

#Эффективное выделение хорошего CV вектора
def cv_extractor(vecs, cv, cvr):
    chain = normalize(diff(expand(vecs)))
    mtrx = lg.inv(chain @ np.transpose(chain))
    cvr = np.transpose(cvr)
    cvs = np.concatenate([cv, cvr], axis = 1)
    cv_mean = np.mean(cvs, axis = 1)
    cv_3d = normalize(mtrx @ chain @ cv_mean)
    cv_good_mean = np.transpose(chain) @ cv_3d
    devs = np.array([lg.norm(cvs[:, i] - cv_good_mean) for i in range(cvs.shape[1])])
    dev = np.sqrt(np.mean(devs ** 2))
    good_cvs = np.transpose(np.array([cvs[:, i] for i in range(cvs.shape[1]) if devs[i] <= 2*dev]))
    cv_mean = np.mean(good_cvs, axis = 1)
    cv_3d = normalize(np.expand_dims(mtrx @ chain @ cv_mean, axis=1))
    return {'center_vec': cv_3d, 'dev': dev  / np.sqrt(cvs.shape[0]),
            'cv': np.repeat(np.transpose(chain) @ cv_3d, cv.shape[1], axis=1)}


#Неправильные предположения относительно симметрии матриц

#Для гетеродимеров, хотя пойдёт и для гомодимеров
def cv_extractor_het(chain1, chain2, cv, cvr): #А тут chain - гениально
    t1 = lg.inv(chain1 @ np.transpose(chain1)) @ chain1
    t2 = np.transpose(chain2) @ lg.inv(chain2 @ np.transpose(chain2))
    #Исходим из симметрии [-1, -1, 1]
    left_sum = t1 @ np.sum(cv, axis = 1)
    right_sum = np.sum(cvr, axis = 0) @ t2
    cv_3d = normalize((left_sum + right_sum) / (cv.shape[1] + cvr.shape[0]))
    cv_est_l = np.transpose(chain1) @ cv_3d
    cv_est_r = np.transpose(cv_3d) @ chain2
    #
    devs_l = np.array([lg.norm(cv[:, i] - cv_est_l) for i in range(cv.shape[1])])
    devs_r = np.array([lg.norm(cvr[i, :] - cv_est_r) for i in range(cvr.shape[0])])
    dev = np.sqrt(np.mean(np.concatenate([devs_l, devs_r]) ** 2))
    good_cv_l = np.transpose(np.array([cv[:, i] for i in range(cv.shape[1]) if devs_l[i] <= 2*dev]))
    good_cv_r = np.array([cvr[i, :] for i in range(cvr.shape[0]) if devs_r[i] <= 2*dev])
    #
    left_sum = t1 @ np.sum(good_cv_l, axis = 1)
    right_sum = np.sum(good_cv_r, axis = 0) @ t2
    cv_3d = normalize((left_sum + right_sum) / (cv.shape[1] + cvr.shape[0]))
    cv_3d = np.expand_dims(cv_3d, axis=1)
    #
    return {'center_vec': cv_3d, 'dev': dev  / np.sqrt(cv.shape[0] + cv.shape[1]), #whatever
            'cv': np.repeat(np.transpose(chain1) @ cv_3d, cv.shape[1], axis=1),
            'cvr': np.repeat(np.transpose(cv_3d) @ chain2, cvr.shape[0], axis=0)}

############################
#MAXIMUM PROTEIN LIKELIHOOD#
############################

#CORRECTED
def prot_likelihood_iter(chain_aim, chain, index, tmtrx = None):
    n = chain.shape[1]
    if tmtrx is None:
        tmtrx = lg.inv(chain_aim[:, index] @ np.transpose(chain_aim[:, index])) @ chain_aim[:, index]
    s = tmtrx @ np.transpose(chain[:, index])

    # s = (np.transpose(s) + s) / 2 #UFFFFFFFFFF ------
    # u, v = lg.eigh(s)
    # #addition
    # if np.prod(u) < 0:                      #REWRITE!!!!!!!!!!!!!!!!
    #     least = np.argmin(np.abs(u))
    #     u[least] = -u[least]
    # #
    # s = v @ np.diag(np.sign(u)) @ np.transpose(v)

    #via SVD
    u, diag, vt = lg.svd(s)
    if lg.det(s) < 0:
        least = np.argmin(np.abs(diag))
        diag[least] = -diag[least]
    s = u @ np.diag(np.sign(diag)) @ vt

    chain_rot = s @ chain
    err = chain_aim - chain_rot
    err = np.array([lg.norm(err[:, i]) for i in range(n)])
    sd = np.sqrt(np.mean(err[index] ** 2))
    return {'sd': sd, 'err': err, 's': s}

def prot_likelihood(chain1, chain2, fullmode = True, border = 0.2, min_points = 15):
    n1 = chain1.shape[1]
    n2 = chain2.shape[1]
    swap = n1 > n2
    if swap:
        chain1, chain2 = chain2, chain1
        n1, n2 = n2, n1
    #chain_1 = chain1
    #chain_2 = chain2
    tmtrx = None
    #var = None
    i0 = 0
    tmtrx = lg.inv(chain1 @ np.transpose(chain1)) @ chain1
    if fullmode:
        var = np.array([np.sum((chain1 - tmtrx @
                        np.transpose(np.take(chain2, range(i,i+n1), 1, mode = 'wrap')) @
                        np.take(chain2, range(i,i+n1), 1, mode = 'wrap')) ** 2) for i in range(n2)])
    else:
        var = np.array([np.sum((chain1 - tmtrx @
                        np.transpose(chain2[:, i:i+n1]) @
                        chain2[:, i:i+n1]) ** 2) for i in range(n2-n1+1)])
    i0 = np.argmin(var)
    chain2 = np.take(chain2, range(i0,i0+n1), 1, mode = 'wrap')
    sd = 100
    i = 0
    index = np.ones(n1, dtype=np.bool8)
    while sd > border: # and not ...
        i += 1
        iter = prot_likelihood_iter(chain1, chain2, index, tmtrx)
        tmtrx = None
        sd = iter['sd']
        index = iter['err'] <= sd
        #chain1 = chain1[:, index]
        #chain2 = chain2[:, index]
        num = np.sum(index)
        if num <= min_points:
            break
        if i >= 100:
            break
    chain2 = iter['s'] @ chain2
    diag = np.array([np.sum(chain1[:, i] * chain2[:, i]) for i in range(n1)])
    return {'s': np.transpose(iter['s']) if swap else iter['s'], 'diag': diag,
            'swap': swap, 'i': i, 'i0': i0, 'sd': sd, 'var': var, 'num': num}

    
def min_points_borders(crop):
    if crop <= 200:
        return 10
    elif crop <= 300:
        return 15
    elif crop <= 400:
        return 20
    else:
        return 35

def sim_metric(res):
    return np.log10((np.mean(res['var']) - np.min(res['var'])) / np.std(res['var']))

def check_goodness(sim, sd):
    return sd <= 0.2 or (sim > 0.8 and sd < 0.3) or (sim > 0.9 and sd < 0.4) or (sim > 1 and sd < 0.5)

def prot_align(chain1, chain2, crop = None):
    if crop is None:
        crop = min(chain1.shape[1], chain2.shape[1])
    res = prot_likelihood(chain1, chain2, min_points=min_points_borders(crop))
    res['sim'] = sim_metric(res)
    return res


############################
#Метод полного предсказания#
############################

#Полное предсказание гомодимера
def full_predict_homo(vecs, cosma_pw, cv, cvr, tol=8e-6, dist=3.4):
    chain = normalize(diff(expand(vecs)))
    
    #Q:
    q_stat = fast_q(vecs, cosma_pw)
    q_v1 = q_stat['vec1']
    q_mtrx = q_stat['Q_ref']
    cosma_pw_ref = np.transpose(chain) @ q_mtrx @ chain 
    q_dev = np.sqrt(np.mean((cosma_pw - cosma_pw_ref) ** 2)) 
    #CV:
    cv_stat = cv_extractor(vecs, cv, cvr)
    cv_3d = cv_stat['center_vec']
    cv_dev = cv_stat['dev']
    #
    prefer_q = q_dev <= cv_dev
    dot = np.sum(q_v1 * cv_3d)
    s_t = np.transpose(q_stat['S'])
    if prefer_q:
        cv_3d = normalize(cv_3d - dot * q_v1)
    else:
        q_v1 = normalize(q_v1 - dot * cv_3d)
        s_t[:, 2] = q_v1.reshape((3))
        q_mtrx = s_t @ np.diag([-1, -1, 1]) @ np.transpose(s_t) #fixed specter
        cosma_pw_ref = np.transpose(chain) @ q_mtrx @ chain 
    #
    vecs2 = rotate_prot(vecs, q_mtrx)
    shift_stat = find_first0(vecs, vecs2, dist, cv_3d, tol)                   #ignore non-C atoms
    vecs2 = shift_prot(vecs2, cv_3d, shift_stat['shift'])
    
    return {'coords': [vecs, vecs2], 's_t': s_t, 'cv_3d': cv_3d, 'q_v1': q_v1,
            'q_dev': q_dev, 'cv_dev': cv_dev, 'prefer_q': prefer_q,
            'cosmapw': cosma_pw_ref, 'cvmtrx': cv_stat['cv'],
            
            '_q_mtrx': q_mtrx, '_shift': cv_3d * shift_stat['shift']} #appended for p&s


# Хм... хм... хм...

#Полное предсказание гетородимера
def full_predict_het(vecs1, vecs2, cosma_pw, cv, cvr, tol=8e-6, dist=3.4):
    chain1 = normalize(diff(expand(vecs1)))
    chain2 = normalize(diff(expand(vecs2)))
    vecs2 = centralize(vecs2)
    vecs2 += np.repeat(get_center(vecs1), vecs2.shape[1], 1)
    
    #Q:
    q_stat = fast_q_het(chain1, chain2, cosma_pw) #hmmmmmmmmmmm......
    q_v1 = q_stat['vec1']
    q_mtrx = q_stat['Q']
    #print(q_stat['eigs'])
    cosma_pw_ref = np.transpose(chain1) @ q_mtrx @ chain2 
    q_dev = np.sqrt(np.mean((cosma_pw - cosma_pw_ref) ** 2)) 
    #CV:
    cv_stat = cv_extractor_het(chain1, chain2, cv, cvr)
    cv_3d = cv_stat['center_vec']
    cv_dev = cv_stat['dev']
    #
    prefer_q = q_dev <= cv_dev
    dot = np.sum(q_v1 * cv_3d)
    s_t = np.transpose(q_stat['S'])
    if prefer_q:
        cv_3d = normalize(cv_3d - dot * q_v1)
    else:
        q_v1 = normalize(q_v1 - dot * cv_3d)
        s_t[:, 2] = q_v1.reshape((3))
        q_mtrx = s_t @ np.diag([-1, -1, 1]) @ np.transpose(s_t) #fixed specter
        cosma_pw_ref = np.transpose(chain1) @ q_mtrx @ chain2 
    #
    vecs2_rot = rotate_prot(vecs2, q_mtrx)
    shift_stat = find_first0(vecs1, vecs2_rot, dist, cv_3d, tol)                          #ignore non-C atoms
    vecs2_rot = shift_prot(vecs2_rot, cv_3d, shift_stat['shift'])
    
    return {'coords': [vecs1, vecs2_rot], 's_t': s_t, 'cv_3d': cv_3d, 'q_v1': q_v1,
            'q_dev': q_dev, 'cv_dev': cv_dev, 'prefer_q': prefer_q,
            'cosmapw': cosma_pw_ref, 'cvmtrx': cv_stat['cv'], 'cvmtrxr': cv_stat['cvr']}


#Удобно...
def degree(vec1, vec2, axis = False):
    cos = np.sum(normalize(vec1) * normalize(vec2))
    if axis:
        cos = np.abs(cos)
    return np.arccos(cos) * 180 / np.pi


####################################################################################################
####################################################################################################

# NEW EXTRACTORS

def norming_l(pt):
    x = pt ** 2
    return np.sqrt(lg.inv(np.transpose(x) @ x) @ np.transpose(x) @ np.ones((pt.shape[0], 1)))[:, 0]

def check_det(mtrx):
    if lg.det(mtrx) < 0:
        mtrx[:, 0] = -mtrx[:, 0]
    return mtrx

def extract_p_from_C_e(cosma): #cosma = p^t @ p
    n = cosma.shape[0]
    u, v = lg.eigh(cosma)
    u = u[n-3:n]
    v = v[:, n-3:n]
    return np.diag(np.sqrt(u)) @ np.transpose(v) #norming_l may be useful here...

def extract_C_es_from_C_p1_p2(cosma): #cosma - p1^t @ p2
    m, n = cosma.shape
    ls2p1t = lg.eigh(cosma @ np.transpose(cosma))[1][:, m-3:m]
    s2p1 = np.diag(norming_l(ls2p1t)) @ np.transpose(ls2p1t)
    C_p1 = np.transpose(s2p1) @ s2p1

    ls1p2t = lg.eigh(np.transpose(cosma) @ cosma)[1][:, n-3:n]
    s1p2 = np.diag(norming_l(ls1p2t)) @ np.transpose(ls1p2t)
    C_p2 = np.transpose(s1p2) @ s1p2

    return (C_p1, C_p2)

def extract_C_e_from_C_a(cosma): #cosma = p^t @ a @ p   det(a)=1
    n = cosma.shape[0]
    lsapt = lg.eigh(np.transpose(cosma) @ cosma)[1][:, n-3:n]
    sap = np.diag(norming_l(lsapt)) @ np.transpose(lsapt)
    return np.transpose(sap) @ sap

###

def extract_p_a_from_C_a(cosma):
    p = extract_p_from_C_e(extract_C_e_from_C_a(cosma))
    a = svd_a(p, p, cosma)
    return (p, a)

def extract_p1_p2_from_C_p1_p1(cosma):
    C_p1, C_p2 = extract_C_es_from_C_p1_p2(cosma)
    p1 = extract_p_from_C_e(C_p1)
    p2_raw = extract_p_from_C_e(C_p2)
    a = svd_a(p1, p2_raw, cosma)
    return (p1, a @ p2_raw)