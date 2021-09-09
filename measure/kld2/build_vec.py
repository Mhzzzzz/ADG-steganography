import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
	logging.info('running %s' % ' '.join(sys.argv))

	for dataset in ['movie', 'news', 'tweet']:
		input_file = '../../data/train_' + dataset + '_nounk'
		model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=0, min_count=5, workers=8)
		model.save(input_file + '.model')
		model.save_word2vec_format(input_file + '.vec')

		sent_file = '../../data/test_' + dataset + '_nounk'
		model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
		model.save_sent2vec_format(sent_file + '.vec')

		files = os.listdir('../../stego/' + dataset)
		for file in files:
			if file.split('.')[-1] == 'txt':
				out_file = '../../stego/' + dataset + '/' + '.'.join(str(file).split('.')[:-1]) + '.vec'
				if not os.path.exists(out_file):
					sent_file = os.path.join('../../stego/' + dataset, file)
					model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
					model.save_sent2vec_format(out_file)

		program = os.path.basename(sys.argv[0])
		logging.info('finished running %s' % program)
