"""
This module implements the Vocab class for converting string to id and back
修改记录: randomly_init_embeddings不再将<ukn>,<padding>初始化为0
load_pretrained_embeddings同样将initial_tokens中所有词随机化而不是赋值为0
暂时取消了随机初始化
"""
import operator
import numpy as np
from hanziconv import HanziConv
from nltk.stem import SnowballStemmer


class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """

    def __init__(self, init_random=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.init_random = init_random

        self.embed_dim = None
        self.embedding_matrix = None

        self.snowballStemmer = SnowballStemmer("english")

        self.pad_token = '<blank>'
        self.spliter_token = '<splitter>'
        self.zero_tokens = [self.pad_token, self.spliter_token]

        self.unk_token = '<unk>'  # 对于测试集中的有的词，可能都不存在与 train 的 oov 可训练的词中，统一作为 unk

        for token in self.zero_tokens:
            self.add(token)

        self.add(self.unk_token)

        # oov的词用于前面，可用于后期词向量可训练
        self.oov_word_start_idx = self.get_id(self.unk_token)
        self.oov_word_end_idx = self.get_id(self.unk_token)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def token_normalize(self, token):
        # url_xxx 的词统一转成 url # TODO 更多的标准化，如中英文标点符号的标准化
        if 'url_' in token:
            token = 'url'

        # 繁体字转换
        token = HanziConv.toSimplified(token)

        return token

    def get_id(self, token, all_unk=False):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            token: a string indicating the word
        Returns:
            an integer
        """
        token = self.token_normalize(token)

        for key in self.translate_word_pipeline(token):
            if key in self.token2id:
                idx = self.token2id[key]
                if all_unk and self.oov_word_start_idx < idx <= self.oov_word_end_idx:
                    idx = self.get_id(self.unk_token)
                return idx

        # 对于 dev 和 test 中的 token，可能不在依据 train 构建的词典中，统一返回 unk
        return self.token2id[self.unk_token]

    def add(self, token, cnt=1):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: a num indicating the count of the token to add, default is 1
        """
        token = self.token_normalize(token)

        for key in self.translate_word_pipeline(token):
            if key in self.token2id:
                idx = self.token2id[key]
                if cnt > 0:
                    self.token_cnt[key] += cnt
                return idx

        # new token
        idx = len(self.id2token)
        self.id2token[idx] = token
        self.token2id[token] = idx
        if cnt > 0:
            self.token_cnt[token] = cnt
        return idx

    def rebuild_add(self, token):
        """
        rebuild vocab, remove text normalization, faster add token
        """
        if token in self.token2id:
            idx = self.token2id[token]
        else:
            idx = len(self.id2token)
            self.id2token[idx] = token
            self.token2id[token] = idx
        return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        left_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] < min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.zero_tokens:
            self.rebuild_add(token)
        self.rebuild_add(self.unk_token)
        for token in left_tokens:
            self.rebuild_add(token)

        # 去掉过滤的词
        for token in filtered_tokens:
            del self.token_cnt[token]

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embedding_matrix = np.random.rand(self.size(), embed_dim)
        if not self.init_random:
            for token in self.zero_tokens:
                self.embedding_matrix[self.get_id(token)] = np.zeros([self.embed_dim])

    def load_pretrained_embed(self, filepath):
        """
        load pretrained embeddings
        """

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float16')

        embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(filepath) if len(o) > 100)

        all_embs = np.stack(list(embeddings_index.values())[:100])
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_dim = all_embs.shape[1]

        return embeddings_index, emb_mean, emb_std, embed_dim

    def translate_word_pipeline(self, key):
        """
        Translate word or correct the misspell to improve the word coverage
        """
        yield key
        yield key.lower()
        yield key.upper()
        yield key.capitalize()
        yield self.snowballStemmer.stem(key)

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.
        Args:
            embeddings_file: A file containing pretrained word embeddings.
        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings into a dictionnary.
        embeddings_index, emb_mean, emb_std, self.embed_dim = self.load_pretrained_embed(embeddings_file)
        print(emb_mean, emb_std)

        # build embedding matrix
        embedding_matrix_tmp = np.random.normal(emb_mean, emb_std, (self.size(), self.embed_dim))

        oov_text_count = 0
        # (oov_word, original_index, count)
        oov_words = []
        not_oov_words = []
        for key, i in self.token2id.items():

            is_oov = True
            for word in self.translate_word_pipeline(key):
                if word in embeddings_index:
                    embedding_matrix_tmp[i] = embeddings_index[word]
                    not_oov_words.append((key, i))
                    is_oov = False
                    break

            if is_oov:
                if key not in self.zero_tokens and key != self.unk_token:
                    oov_text_count += self.token_cnt[key]
                    oov_words.append((key, i, self.token_cnt[key]))

        oov_count = len(oov_words)
        print("Missed words: ", oov_count)
        print('Found embeddings for {:.6%} of vocab'.format((self.size() - oov_count) / self.size()))
        print('Found embeddings for {:.6%} of all text'.format(oov_text_count / sum(self.token_cnt.values())))
        print('Save out of vocabulary words:')
        oov_words = sorted(oov_words, key=operator.itemgetter(2))[::-1]
        with open('logs/oov_words.txt', 'w') as oov_writer:
            oov_writer.writelines([oov[0] + '\t' + str(oov[2])+ '\n' for oov in oov_words])

        print('Move oov words ahead and rebuild the token x id map')
        self.token2id = {}
        self.id2token = {}

        # zero words
        for token in self.zero_tokens:
            self.rebuild_add(token)

        # oov words
        self.rebuild_add(self.unk_token)
        for oov in oov_words:
            self.rebuild_add(oov[0])
        self.oov_word_start_idx = self.get_id(self.unk_token)  # oov 词开始下标
        self.oov_word_end_idx = self.get_id(oov_words[-1][0])  # oov 词结束下标

        # not oov words
        for token in not_oov_words:
            self.rebuild_add(token[0])

        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.size(), self.embed_dim))

        # zero words vector
        for zero_token in self.zero_tokens:
            self.embedding_matrix[self.get_id(zero_token)] = np.zeros(shape=self.embed_dim)

        # random oov words vector

        # set vectors for not oov words
        for not_oov_word in not_oov_words:
            self.embedding_matrix[self.get_id(not_oov_word[0])] = embedding_matrix_tmp[not_oov_word[1]]

        print('Final vocabulary size:', self.size())
        print(f'trainable oov words start from {self.oov_word_start_idx} to {self.oov_word_end_idx}')

    def convert_to_ids(self, tokens, all_unk=False):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
            all_unk: 所有oov的词是否映射到 <unk>, 默认为 False
        Returns:
            a list of ids
        """
        vec = [self.get_id(token, all_unk) for token in tokens]
        return vec
