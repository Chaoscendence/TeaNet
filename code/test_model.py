import os
import numpy as np
import torch

def test_TEA(splitted_data, settings, save_path, num):
    file = open(save_path + '/acc.txt', 'w')
    saved_net = torch.load(os.path.join(save_path + '/', 'best_D.tar'))
    net = saved_net['net']

    device = torch.device("cuda:" + str(num)  if settings['cuda'] else "cpu")
    net = net.to(device)
    net.eval()
    datasets = ['test']
    for ds in datasets:
        X_test, Y_test= splitted_data[ds][0], splitted_data[ds][1]

        acc = 0.0
        n = 0

        num_test = X_test.shape[0]
        pred_output = []
        for i in range(num_test):
            tmp = torch.from_numpy(X_test[i].reshape(1,1,-1)).float().to(device)
            y = net(tmp)
            _, y = y.data.max(dim=1)
            pred_output.append(y.cpu())
            pred_output.append(Y_test[i])
            n += 1
            if y.cpu() == Y_test[i]:
                acc += 1

        file.write(ds + '    ' + str(acc/n) + '\n')
        pred_output = np.array(pred_output)

        np.savetxt(save_path + '/' + ds + '_predict.txt', pred_output)

    file.close()
