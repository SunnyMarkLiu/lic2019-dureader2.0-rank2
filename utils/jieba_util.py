#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/4/11 18:35
"""
from jieba.analyse.tfidf import TFIDF


class WordSegmentPOSKeywordExtractor(TFIDF):

    def extract_sentence(self, sentence, keyword_ratios=0.8):
        """
        Extract keywords from sentence using TF-IDF algorithm.
        Parameter:
            - keyword_ratios: return how many top keywords. `None` for all possible words.
        """
        words = self.postokenizer.cut(sentence)
        freq = {}

        seg_words = []
        pos_words = []
        for w in words:
            wc = w.word
            seg_words.append(wc)
            pos_words.append(w.flag)

            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[wc] = freq.get(wc, 0.0) + 1.0

        total = sum(freq.values())
        for k in freq:
            freq[k] *= self.idf_freq.get(k, self.median_idf) / total

        tags = sorted(freq, key=freq.__getitem__, reverse=True)
        top_k = int(keyword_ratios * len(seg_words))
        tags = tags[:top_k]

        key_words = [int(word in tags) for word in seg_words]

        return seg_words, pos_words, key_words
