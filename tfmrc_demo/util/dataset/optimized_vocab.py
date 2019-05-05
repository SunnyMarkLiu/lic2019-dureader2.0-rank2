"""
This module implements the Vocab class for converting string to id and back
"""
import re
import logging
import operator
import numpy as np
from tqdm import tqdm
from hanziconv import HanziConv
from nltk.stem import SnowballStemmer


class LoggerMixin(object):
    @property
    def logger(self):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        return logging.getLogger(component)


class Vocab(LoggerMixin):
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

        self.add(self.unk_token)

        for token in self.zero_tokens:
            self.add(token)

        # oov的词用于前面，可用于后期词向量可训练
        self.oov_word_end_idx = self.get_id(self.unk_token)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def token_normalize(self, token):
        # url_xxx 的词统一转成 url # TODO 更多的标准化
        # if token == ' ':
        #     return self.pad_token

        if 'url_' in token:
            token = 'url'

        # if token.lower() in {'win10', 'win7', 'windows10', 'windows7', 'windows8', 'window7',
        #                      'windows10windows', 'windows2000', 'windows7windows', 'windowxp',
        #                      'windownt', 'window10', 'windowswindows', 'windows98', 'windows9x'}:
        #     return 'Windows'
        #
        # # iphone
        # token = re.sub('iphone(.+)', 'iphone', token.lower())

        # 繁体字转换
        token = HanziConv.toSimplified(token)
        # # 数字和标点符号组合的词的标准化
        # token = re.sub(r'(\d+)\.', '\g<1>', token)
        # token = re.sub(r'\.(\d+)', '\g<1>', token)
        # 唿 -> 呼
        # token = re.sub('唿', '呼', token)
        # token = re.sub(
        #     r'(第*)(有|多|几|半|一|两|二|三|四|五|六|七|八|九|十)(种|个|次|款|篇|天|步|年|大|条|方|位|键|份|项|周|层|只|套|名|句|件|台|部|页|段|把|片|小时|遍|颗|根|批|张|分|性|点点|场|分钟|组|堆|本|圈|季|笔|群|斤|日|支|排|章|所|股|门|首|代|号|生|点|辆|轮|瓶|声|杯|列|座|集)',
        #     '\g<2>', token)

        return token

    def get_id(self, token, all_unk=False):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            token: a string indicating the word
            all_unk: 所有oov的词是否映射到 <unk>, 默认为 False
        Returns:
            an integer
        """
        token = self.token_normalize(token)

        for key in self.translate_word_pipeline(token):
            if key in self.token2id:
                idx = self.token2id[key]
                if all_unk and idx <= self.oov_word_end_idx:
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
        self.rebuild_add(self.unk_token)
        for token in self.zero_tokens:
            self.rebuild_add(token)
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
        embeddings_index = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_size, embed_dim = map(int, f.readline().strip().split(" "))

            for _ in tqdm(range(vocab_size)):
                lists = f.readline().rstrip().split(" ")
                word = lists[0]
                vector = np.asarray(list(map(float, lists[1:])), dtype='float16')
                embeddings_index[word] = vector

        sample_embs = np.stack(list(embeddings_index.values())[:1000])
        emb_mean, emb_std = sample_embs.mean(), sample_embs.std()

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
        self.logger.info('pretrained embeddings mean: {}, std: {}, calc from top 1000 words'.format(emb_mean, emb_std))

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
        self.logger.info("Missed words: {}".format(oov_count))
        self.logger.info('Found embeddings for {:.6%} of vocab'.format((self.size() - oov_count) / self.size()))
        self.logger.info(
            'Found embeddings for {:.6%} of all text'.format(1 - oov_text_count / sum(self.token_cnt.values())))
        self.logger.info('Save out of vocabulary words to logs/oov_words.txt')
        oov_words = sorted(oov_words, key=operator.itemgetter(2))[::-1]
        with open('logs/oov_words_{}.txt'.format(embeddings_file.split('/')[-1]), 'w') as oov_writer:
            oov_writer.writelines([oov[0] + '\t' + str(oov[2]) + '\n' for oov in oov_words])

        self.logger.info('Move oov words ahead and rebuild the token x id map')
        self.token2id = {}
        self.id2token = {}

        # oov words
        self.rebuild_add(self.unk_token)
        for oov in oov_words:
            self.rebuild_add(oov[0])
        self.oov_word_end_idx = self.get_id(oov_words[-1][0])  # oov 词结束下标

        # zero words
        for token in self.zero_tokens:
            self.rebuild_add(token)

        # not oov words
        for token in not_oov_words:
            self.rebuild_add(token[0])

        # build embedding matrix, random oov words vector
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.size(), self.embed_dim))

        # zero words vector
        for zero_token in self.zero_tokens:
            self.embedding_matrix[self.get_id(zero_token)] = np.zeros(shape=self.embed_dim)

        # set vectors for not oov words
        for not_oov_word in not_oov_words:
            self.embedding_matrix[self.get_id(not_oov_word[0])] = embedding_matrix_tmp[not_oov_word[1]]

        self.logger.info('Final vocabulary size: {}'.format(self.size()))
        self.logger.info('trainable oov words start from 0 to {}'.format(self.oov_word_end_idx))

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
