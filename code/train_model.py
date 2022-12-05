import numpy as np
import torch
import os
import torch.utils.data as data
from models.unet import Unet
from utils.utils import save_checkpoint
from torch.optim import lr_scheduler
import torch.nn as nn
import os
from models.unet import Unet
from models.lenet_ir import LeNet
from models.loss_func import MyLossFunc
import math
from itertools import cycle

# torch.device object used throughout this script
_g_device = None

# This settings will affect all the functions below
_g_settings = None

# predict_save = None

def set_config(settings, num):
    global _g_settings
    _g_settings = settings
    global _g_device
    _g_device = torch.device("cuda:" + str(num) if _g_settings['cuda'] else "cpu")


def train_TEA(splitted_data, save_path, logger):
    epochs = _g_settings['epochs']
    save_path = save_path + '/'
    batch_size = _g_settings['batch_size']
    verbose = _g_settings['verbose']
    ratio = _g_settings['loss_ratio']
    classifier = LeNet(settings=_g_settings)
    generator = Unet(settings=_g_settings)
    criterion_G = MyLossFunc()
    generator = generator.to(_g_device)
    classifier = classifier.to(_g_device)
    criterion_D = nn.CrossEntropyLoss()
    criterion_G = criterion_G.to(_g_device)
    criterion_D = criterion_D.to(_g_device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=_g_settings['lr_base_G'])
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.75)

    optimizer_D = torch.optim.Adam(classifier.parameters(), lr=_g_settings['lr_base_D'])
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.75)

    X_tr, y_tr, c_tr = splitted_data['train'][0], splitted_data['train'][2], splitted_data['train'][1]
    X_d_tr, c_d_tr = splitted_data['default_train'][0], splitted_data['default_train'][1] 
    X_va, y_va, c_va = splitted_data['valid'][0], splitted_data['valid'][2], splitted_data['valid'][1]
    X_te, c_te = splitted_data['test'][0], splitted_data['test'][1]

    X_tr = torch.Tensor(X_tr)
    y_tr = torch.Tensor(y_tr)
    c_tr = torch.Tensor(c_tr)
    X_d_tr = torch.Tensor(X_d_tr)
    c_d_tr = torch.Tensor(c_d_tr)
    X_va = torch.Tensor(X_va)
    y_va = torch.Tensor(y_va)
    c_va = torch.Tensor(c_va)
    X_te = torch.Tensor(X_te)
    c_te = torch.Tensor(c_te)

    print('#Training_Samples: {0}, #Valid_Samples: {1}, #Test_Samples: {2}'.
        format(y_tr.size()[0], y_va.size()[0], c_te.size()[0]))

    # best_valid_g_loss = float('inf')
    best_valid_d_loss = float('inf')
    best_test = -float('inf')
    best_valid_d_epoch = -1
    g_loss_epochs = []
    d_loss_epochs = []
    valid_loss_g_epochs = []
    valid_loss_d_epochs = []
    tmp = True
    for epoch in range(epochs):
        generator.train()
        classifier.train()
        g_loss_epoch, d_loss_epoch = run_model_one_epoch(training_G=tmp, training_D=not tmp, generator=generator, episodes=[X_tr, y_tr, c_tr, X_d_tr, c_d_tr],
                                               batch_size=batch_size, criterion_G=criterion_G,
                                               criterion_D=criterion_D,classifier=classifier,
                                               optimizer_G=optimizer_G, optimizer_D=optimizer_D, ratio=ratio)
        if tmp:
            scheduler_G.step()
        else:
            scheduler_D.step()
        tmp = not tmp
        g_loss_epochs += [g_loss_epoch]
        d_loss_epochs += [d_loss_epoch]

        np.save(os.path.join(save_path,'train_g_loss'),np.array(g_loss_epochs))
        np.save(os.path.join(save_path,'train_d_loss'),np.array(d_loss_epochs))

        with torch.no_grad():
            generator.eval()
            classifier.eval()
            valid_loss_g_epoch, valid_loss_d_epoch = valid_model_one_epoch(episodes=[X_va, y_va, c_va], generator=generator, 
                                                batch_size=batch_size, criterion_G=criterion_G, criterion_D=criterion_D,classifier=classifier, ratio=ratio)
            valid_loss_g_epochs += [valid_loss_g_epoch]
            valid_loss_d_epochs += [valid_loss_d_epoch]

            is_better = valid_loss_d_epoch < best_valid_d_loss

            save_checkpoint({
                'epoch': iter,
                'net': generator,
                'best_valid_loss': valid_loss_g_epoch,
                'optimizer': optimizer_G.state_dict(),
            }, is_better, best_filename=os.path.join(save_path, 'best_G.tar'))
            if is_better:
                best_valid_d_loss = valid_loss_d_epoch
                best_valid_d_epoch = epoch + 1
            save_checkpoint({
                'epoch' : iter,
                'net' : classifier,
                'best_valid_loss' : valid_loss_d_epoch,
                'optimizer' : optimizer_D.state_dict(),
            }, is_better, best_filename=os.path.join(save_path, 'best_D.tar'))
        with torch.no_grad():
            classifier.eval()
            acc_te = test_model_one_epoch(episodes=[X_te, c_te], batch_size=batch_size, classifier=classifier)
            best_test = acc_te if acc_te > best_test else best_test

        if verbose:
            logger.info('epoch: {} g_loss: {:.8f}, d_loss: {:.8f}, Valid_g_loss: {:.8f}, Valid_d_loss: {:.8f}, acc_te: {:.8f}, best_test: {:.8f}, best_epoch: {}'.format(epoch + 1, g_loss_epoch, d_loss_epoch, valid_loss_g_epoch, valid_loss_d_epoch, acc_te, best_test, best_valid_d_epoch))
    np.save(os.path.join(save_path, 'valid_g_loss'), np.array(valid_loss_g_epochs))
    np.save(os.path.join(save_path, 'valid_d_loss'), np.array(valid_loss_d_epochs))

def run_model_one_epoch(training_G, training_D, generator, episodes, batch_size, criterion_G, criterion_D, classifier,
                                               optimizer_G, optimizer_D, ratio):
    d_losses = []
    g_losses = []
    dataset = data.TensorDataset(episodes[0], episodes[1], episodes[2])
    loader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=
        batch_size)
    dataset_d = data.TensorDataset(episodes[3], episodes[4])
    loader_d = data.DataLoader(dataset=dataset_d, shuffle=True, batch_size=math.ceil(len(episodes[3])/math.ceil(len(episodes[0])/batch_size)))
    for _, tmp in enumerate(zip(loader, cycle(loader_d))):
        X_batch, y_batch, c_batch = tmp[0]
        X_tmp, c_tmp = tmp[1] 
        y_batch = y_batch.unsqueeze(1)
        X_batch, y_batch, X_tmp = X_batch.to(_g_device), y_batch.to(_g_device), X_tmp.to(_g_device)
        X_batch = X_batch.unsqueeze(1)
        X_batch = X_batch.requires_grad_()
        X_tmp = X_tmp.unsqueeze(1)
        X_tmp = X_tmp.requires_grad_()
        y_prob = generator(X_batch)

        y_prob_tmp = torch.cat((y_prob, X_tmp), 0)

        c_y_prob = classifier(y_prob_tmp)
        c_batch = torch.cat((c_batch, c_tmp),0)
        c_batch = c_batch.long()
        c_batch = c_batch.to(_g_device)

        d_loss = criterion_D(c_y_prob, c_batch)
        loss = (1 - ratio) * criterion_G(y_prob, y_batch) + ratio * d_loss
        g_losses.append(loss.item())
        d_losses.append(d_loss.item())
        if training_G:
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
        if training_D:
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
    return np.mean(g_losses), np.mean(d_losses)


def test_model_one_epoch(episodes, batch_size, classifier):
    dataset = data.TensorDataset(episodes[0], episodes[1])
    loader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=
        batch_size)
    acc = 0.0
    n = 0
    for _, (X_batch, c_batch) in enumerate(loader):
        X_batch = X_batch.unsqueeze(1)
        X_batch = X_batch.requires_grad_()
        X_batch = X_batch.to(_g_device)

        c_y_prob = classifier(X_batch)
        c_batch = c_batch.long()
        c_batch = c_batch.to(_g_device)
        
        n += c_batch.shape[0]
        acc += (c_y_prob.argmax(dim = 1) == c_batch).sum().cpu().item()

    return acc / n


def valid_model_one_epoch(generator, episodes, batch_size, criterion_D, criterion_G, classifier, ratio):
    d_losses = []
    g_losses = []
    dataset = data.TensorDataset(episodes[0], episodes[1], episodes[2])
    loader = data.DataLoader(dataset=dataset, shuffle=True, batch_size=
        batch_size)
    acc = 0.0
    n = 0
    for _, (X_batch, y_batch, c_batch) in enumerate(loader):
        X_batch = X_batch.unsqueeze(1)
        y_batch = y_batch.unsqueeze(1)
        X_batch, y_batch = X_batch.to(_g_device),y_batch.to(_g_device)
        if False:
            y_prob = generator(X_batch)
        else:
            y_prob = y_batch
        c_y_prob = classifier(y_prob)
        c_batch = c_batch.long()
        c_batch = c_batch.to(_g_device)

        d_loss = criterion_D(c_y_prob, c_batch)
        loss = (1 - ratio) * criterion_G(y_prob, y_batch) + ratio * d_loss
        # loss = d_loss
        g_losses.append(loss.item())
        d_losses.append(d_loss.item())
        n += c_batch.shape[0]
        acc += (c_y_prob.argmax(dim = 1) == c_batch).sum().cpu().item()

    print('valid_acc %.3f' % (acc/n))
    return np.mean(g_losses), np.mean(d_losses)
