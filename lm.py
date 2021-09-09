import torch
import torch.nn as nn


class LM(nn.Module):
	def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate):
		super(LM, self).__init__()
		self._cell = cell

		self.embedding = nn.Embedding(vocab_size, embed_size)
		if cell == 'rnn':
			self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
		elif cell == 'gru':
			self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
		elif cell == 'lstm':
			self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, dropout=dropout_rate)
		else:
			raise Exception('no such rnn cell')

		self.output_layer = nn.Linear(hidden_dim, vocab_size)
		self.log_softmax = nn.LogSoftmax(dim=2)

	def forward(self, x, logits=False):
		x = x.long()
		_ = self.embedding(x)
		_ = _.permute(1, 0, 2)
		h_all, __ = self.rnn(_)
		h_all = h_all.permute(1, 0, 2)
		_ = self.output_layer(h_all)
		if logits:
			return _
		else:
			return self.log_softmax(_)

	def sample(self, x):
		log_prob = self.forward(x)
		prob = torch.exp(log_prob)[:, -1, :]
		# p, i = prob.sort(descending=True)
		# self.p = p
		prob[:, 1] = 0
		prob = prob / prob.sum()
		return torch.multinomial(prob, 1)
