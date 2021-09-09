import numpy as np
import os
import json


def klc(mean1, std1, mean2, std2):
    var1 = np.power(std1, 2) + 0.0000001
    var2 = np.power(std2, 2) + 0.0000001
    return 0.5 * np.sum(np.log(var2) - np.log(var1) - 1
                        + np.true_divide(var1, var2) + np.true_divide(np.power((mean2 - mean1), 2), var2))


if __name__ == '__main__':
    result = {}
    for dataset in ['movie', 'news', 'tweet']:
        result[dataset] = {}
        # mean and std of training set
        ftest = open('../../data/test_' + dataset + '_nounk.vec', encoding='utf8')
        flist = list(ftest)
        vec_test = []
        for index in range(1, len(flist), 1):
            vec_test += flist[index].strip().split()[1:]

        vec_test = np.array(vec_test).astype(float).reshape(-1, 100)
        mean1 = np.mean(vec_test, axis=0)
        std1 = np.std(vec_test, axis=0)

        files = os.listdir('../../stego/' + dataset)
        for file in files:
            if file.split('.')[-1] == 'txt':
                try:
                    a = result[dataset]['.'.join(str(file).split('.')[:-1])]
                except:
                    vec_file = '../../stego/' + dataset + '/' + '.'.join(str(file).split('.')[:-1]) + '.vec'
                    with open(vec_file, 'r', encoding='utf8') as f:
                        flist = list(f)
                        vec = []
                        for index in range(1, len(flist), 1):
                            vec += flist[index].strip().split()[1:]
                        vec_np = np.array(vec).astype(float).reshape(-1, 100)
                    mean2 = np.mean(vec_np, axis=0)
                    std2 = np.std(vec_np, axis=0)
                    kl = klc(mean1, std1, mean2, std2)
                    result[dataset]['.'.join(str(file).split('.')[:-1])] = kl

    # write files
    with open('kld2.json', 'w', encoding='utf8') as f:
        json.dump(result, f)
