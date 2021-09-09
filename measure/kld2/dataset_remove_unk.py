import utils


if __name__ == '__main__':
    for DATASET in ['movie', 'news', 'tweet']:
        WORD_DROP = 10
        MIN_LEN = 5
        MAX_LEN = 200

        data_path = '../../data/' + DATASET + '2020.txt'
        train_path = '../../data/train_' + DATASET
        test_path = '../../data/test_' + DATASET
        vocabulary = utils.Vocabulary(
            data_path,
            max_len=MAX_LEN,
            min_len=MIN_LEN,
            word_drop=WORD_DROP
        )

        test = utils.Corpus(test_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)
        with open(test_path + '_nounk', 'w', encoding='utf8') as f:
            for sentence in test.corpus:
                if 1 not in sentence:
                    f.write(' '.join([vocabulary.i2w[_] for _ in sentence if _ not in [0, 2, 3]]) + '\n')

        train = utils.Corpus(train_path, vocabulary, max_len=MAX_LEN, min_len=MIN_LEN)
        with open(train_path + '_nounk', 'w', encoding='utf8') as f:
            for sentence in train.corpus:
                if 1 not in sentence:
                    f.write(' '.join([vocabulary.i2w[_] for _ in sentence if _ not in [0, 2, 3]]) + '\n')
