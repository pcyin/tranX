# coding=utf-8


import math
import numpy as np
from torch.autograd import Variable


class GloveHelper(object):
    def __init__(self, glove_file):
        self.glove_file = glove_file
        embeds = np.zeros((5000, 100), dtype='float32')
        for i, (word, embed) in enumerate(self.embeddings):
            if i == 5000: break
            embeds[i] = embed

        self.mean = np.mean(embeds)
        self.std = np.std(embeds)

    @property
    def embeddings(self):
        with open(self.glove_file, 'r') as f:
            for line in f:
                tokens = line.split()
                word, embed = tokens[0], np.array([float(tok) for tok in tokens[1:]])
                yield word, embed

    def emulate_embeddings(self, shape):
        samples = np.random.normal(self.mean, self.std, size=shape)

        return samples

    def load_to(self, embed_layer, vocab):
        new_tensor = embed_layer.weight.data.new
        word_ids = set(range(embed_layer.num_embeddings))
        for word, embed in self.embeddings:
            if word in vocab:
                word_id = vocab[word]
                word_ids.remove(word_id)
                embed_layer.weight[word_id].data = new_tensor(embed)

        word_ids = list(word_ids)
        embed_layer.weight[word_ids].data = new_tensor(self.emulate_embeddings(shape=(len(word_ids), embed_layer.embedding_dim)))

    @property
    def words(self):
        with open(self.glove_file, 'r') as f:
            for line in f:
                tokens = line.split()
                yield tokens[0]


def batch_iter(examples, batch_size, shuffle=False):
    batch_num = int(math.ceil(len(examples) / float(batch_size)))
    index_array = list(range(len(examples)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_examples = [examples[idx] for idx in indices]

        yield batch_examples


def get_parser_class(lang):
    if lang in ['python', 'lambda_dcs', 'prolog', 'python3']:
        from model.parser import Parser
        return Parser
    elif lang == 'wikisql':
        from model.wikisql.parser import WikiSqlParser
        return WikiSqlParser
    else:
        raise ValueError('unknown parser class for %s' % lang)
