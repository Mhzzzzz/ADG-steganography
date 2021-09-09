import collections
import numpy as np


class Vocabulary(object):
	def __init__(self, data_path, max_len=200, min_len=5, word_drop=5, encoding='utf8'):
		if type(data_path) == str:
			data_path = [data_path]
		self._data_path = data_path
		self._max_len = max_len
		self._min_len = min_len
		self._word_drop = word_drop
		self._encoding = encoding
		self.token_num = 0
		self.vocab_size_raw = 0
		self.vocab_size = 0
		self.w2i = {}
		self.i2w = {}
		self.start_words = []
		self._build_vocabulary()

	def _build_vocabulary(self):
		self.w2i['_PAD'] = 0
		self.w2i['_UNK'] = 1
		self.w2i['_BOS'] = 2
		self.w2i['_EOS'] = 3
		self.i2w[0] = '_PAD'
		self.i2w[1] = '_UNK'
		self.i2w[2] = '_BOS'
		self.i2w[3] = '_EOS'
		words_all = []
		start_words = []
		for data_path in self._data_path:
			with open(data_path, 'r', encoding=self._encoding) as f:
				sentences = f.readlines()
			for sentence in sentences:
				_ = sentence.split()
				if (len(_) >= self._min_len) and (len(_) <= self._max_len):
					words_all.extend(_)
					start_words.append(_[0])
		self.token_num = len(words_all)
		word_distribution = sorted(collections.Counter(words_all).items(), key=lambda x: x[1], reverse=True)
		self.vocab_size_raw = len(word_distribution)
		for (word, value) in word_distribution:
			if value > self._word_drop:
				self.w2i[word] = len(self.w2i)
				self.i2w[len(self.i2w)] = word
		self.vocab_size = len(self.i2w)
		start_word_distribution = sorted(collections.Counter(start_words).items(), key=lambda x: x[1], reverse=True)
		self.start_words = [_[0] for _ in start_word_distribution]


class Corpus(object):
	def __init__(self, data_path, vocabulary, max_len=200, min_len=5):
		if type(data_path) == str:
			data_path = [data_path]
		self._data_path = data_path
		self._vocabulary = vocabulary
		self._max_len = max_len
		self._min_len = min_len
		self.corpus = []
		self.corpus_length = []
		self.labels = []
		self.sentence_num = 0
		self.max_sentence_length = 0
		self.min_sentence_length = 0
		self._build_corpus()

	def _build_corpus(self):
		def _transfer(word):
			try:
				return self._vocabulary.w2i[word]
			except:
				return self._vocabulary.w2i['_UNK']
		label = -1
		for data_path in self._data_path:
			label += 1
			with open(data_path, 'r', encoding='utf8') as f:
				sentences = f.readlines()
			for sentence in sentences:
				sentence = sentence.split()
				if (len(sentence) >= self._min_len) and (len(sentence) <= self._max_len):
					sentence = ['_BOS'] + sentence + ['_EOS']
					self.corpus.append(list(map(_transfer, sentence)))
					self.labels.append(label)
		self.corpus_length = [len(i) for i in self.corpus]
		self.max_sentence_length = max(self.corpus_length)
		self.min_sentence_length = min(self.corpus_length)
		self.sentence_num = len(self.corpus)


def split_corpus(data_path, train_path, test_path, max_len=200, min_len=5, ratio=0.8, seed=0, encoding='utf8'):
	with open(data_path, 'r', encoding=encoding) as f:
		sentences = f.readlines()
	sentences = [_ for _ in filter(lambda x: x not in [None, ''], sentences)
	             if len(_.split()) <= max_len and len(_.split()) >= min_len]
	np.random.seed(seed)
	np.random.shuffle(sentences)
	train = sentences[:int(len(sentences) * ratio)]
	test = sentences[int(len(sentences) * ratio):]
	with open(train_path, 'w', encoding='utf8') as f:
		for sentence in train:
			f.write(sentence)
	with open(test_path, 'w', encoding='utf8') as f:
		for sentence in test:
			f.write(sentence)


class Generator(object):
	def __init__(self, data, vocabulary=None):
		self._data = np.array(data)
		self._vocabulary = vocabulary

	def _padding(self, batch_data):
		assert self._vocabulary is not None
		max_length = max([len(i) for i in batch_data])
		for i in range(len(batch_data)):
			batch_data[i] += [self._vocabulary.w2i["_PAD"]] * (max_length - len(batch_data[i]))
		return np.array(list(batch_data))

	def build_generator(self, batch_size, shuffle=True, padding=True):
		indices = list(range(len(self._data)))
		if shuffle:
			np.random.shuffle(indices)
		while True:
			batch_indices = indices[0:batch_size]               # 产生一个batch的index
			indices = indices[batch_size:]                      # 去掉本次index
			if len(batch_indices) == 0:
				return True
			batch_data = self._data[batch_indices]
			if padding:
				batch_data = self._padding(batch_data)
			yield batch_data
