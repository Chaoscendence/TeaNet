import numpy as np
from numpy.random import permutation

def dataset_divide(scheme, data):
    """Create training and test dataset
       leave one out
    """
    spectras = data['train_valid'][0]
    labels = data['train_valid'][1]
    ridx = permutation(range(len(spectras)))
    t_v = int(round(scheme * len(spectras)))
    t_v = ridx[0 : t_v]
    v_t = [x for x in range(len(spectras))
           if x not in t_v]

    X_train = spectras[t_v.tolist(), :]
    Y_train = labels[t_v.tolist()]
    X_valid = spectras[v_t, :]
    Y_valid = labels[v_t]
    splitted_data = {
        'train': (X_train, Y_train),
        'valid': (X_valid, Y_valid),
        'test': data['test']
    }
    return splitted_data

def mask_each_dataset(data, scheme, num):
    spectras = data[0]
    labels = data[1]

    length = None
    if scheme[0] > 1:
        length = int(scheme[0])

    masked_data = []
    ori_data = []
    maked_label = []
    
    for i in range(len(spectras)):
        for _ in range(num):    
            if scheme[0] <= 1.0:
                length = int(round(len(spectras[i]) * (scheme[1] - scheme[0]) * np.random.rand())) + int(round(len(spectras[i]) * scheme[0]))
            strt = np.random.randint(0, len(spectras[i]) - length + 1)
            tmp = spectras[i].copy()
            ori_data.append(spectras[i])
            for m in range(length):
                tmp[strt + m] = float(-1)
            masked_data.append(tmp)
            maked_label.append(labels[i])
    return np.array(masked_data), np.array(maked_label), np.array(ori_data)
def dataset_mask(data, scheme, num, split = None):
    
    if 'train_valid' in data:
        train_valid_masked, train_valid_y, train_valid_spec = mask_each_dataset(data['train_valid'], scheme, num)
    else:
        train_masked, train_y, train_spec = mask_each_dataset(data['train'], scheme, num)
        valid_masked, valid_y, valid_spec = mask_each_dataset(data['valid'], scheme, 1)
    if split != None:
        ridx = permutation(range(len(train_valid_masked)))
        t_v = int(round(split * len(train_valid_masked)))
        t_v = ridx[0 : t_v]
        v_t = [x for x in range(len(train_valid_masked))
           if x not in t_v]
        train_masked = train_valid_masked[t_v.tolist(), :]
        train_y = train_valid_y[t_v.tolist()]
        train_spec = train_valid_spec[t_v.tolist(), :]
        valid_spec = train_valid_spec[v_t, :]
        valid_masked = train_valid_masked[v_t, :]
        valid_y = train_valid_y[v_t]
    splitted_data = {
        'default_train': (train_spec, train_y) if split != None else data['train'],
        'default_valid': (valid_spec, valid_y) if split != None else data['valid'],
        'train': (train_masked, train_y, train_spec),
        'valid': (valid_masked, valid_y, valid_spec),
        'test' : data['test']
    }
    return splitted_data
