import os
import json
import numpy as np


if __name__ == '__main__':
    result = {}
    for dataset in ['movie', 'news', 'tweet']:
        result[dataset] = {}
        files = os.listdir('../../stego/' + dataset)
        for file in files:
            if file.split('.')[-1] == 'txt' and \
                    os.path.exists(os.path.join('../../stego/' + dataset, '.'.join(str(file).split('.')[:-1]) + '.bit')):
                path = os.path.join('../../stego/' + dataset, '.'.join(str(file).split('.')[:-1]))

                with open(path + '.txt', encoding='utf8') as f:
                    sentences = f.readlines()
                sentences = [_.strip() for _ in sentences]
                sentences = list(map(lambda x: x.split(), sentences))

                with open(path + '.bit', encoding='utf8') as f:
                    bits = f.readlines()
                bits = [_.strip() for _ in bits]

                em = []
                for i in range(len(sentences)):
                    em.append(len(bits[i]) / (len(sentences[i]) - 1))        # drop start word

                result[dataset]['.'.join(str(file).split('.')[:-1])] = {}
                result[dataset]['.'.join(str(file).split('.')[:-1])]['mean'] = np.mean(em)
                result[dataset]['.'.join(str(file).split('.')[:-1])]['std'] = np.std(em, ddof=1)

    # write files
    with open('er.json', 'w', encoding='utf8') as f:
        json.dump(result, f)
