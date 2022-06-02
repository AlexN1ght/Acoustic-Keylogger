import string
import re

import nltk
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('reuters')
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import reuters

from nltk.probability import FreqDist
from nltk import ngrams

from .language_helpers import *

def clear_data(X, matched_indexes, new_lbls):
    lbls = list(new_lbls)
    X_cleared = np.zeros((len(matched_indexes), X.shape[1]))
    pos = 0
    for i in matched_indexes:
        if i == len(X):
            lbls.pop(pos)
            continue
        X_cleared[pos] = X[i]
        pos += 1
    return X_cleared, list(new_lbls)

def normalize(a):
    max_item = max(a)
    for i in range(len(a)):
        a[i] /= max_item
    return a

class BigramPredictor(object):
    def __init__(self, word_list_dataset):
        self.word_list_dataset = word_list_dataset

        print("Gathering data from corpus")
        self.sents = []
        for fileid in brown.fileids():
            self.sents += list(brown.sents(fileid))
        for fileid in webtext.fileids():
            self.sents += list(webtext.sents(fileid))
        for fileid in gutenberg.fileids():
            self.sents += list(gutenberg.sents(fileid))
        for fileid in reuters.fileids():
            self.sents += list(reuters.sents(fileid))

        print("Data Len:", len(self.sents), "sentences")


        string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—'+'--'
        self.removal_list = list(string.punctuation)+ ['lt','rt', '``', "''"]


        print("Generating unigrams and bigrams")

        self.unigram=[]
        self.bigram=[]
        tokenized_text=[]
        for sentence in self.sents:
            sentence = list(map(lambda x:x.lower(), sentence))
            for word in sentence:
                if word in self.removal_list:
                    sentence.remove(word)
                else:
                    self.unigram.append(word)
        
            tokenized_text.append(sentence)
            self.bigram.extend(list(ngrams(sentence, 2,pad_left=True, pad_right=True)))

        print("Calculating bigram FreqDist")
        self.freq_bi = FreqDist(self.bigram)
        print("Bigram Len:", len(self.freq_bi))

    
    def Viterbi(self, text, max_corr_a=0.2, E=get_unified_cond_prob_matrix(0.5)):
        words = re.split('( |\.|,|\$)', text)
        n = 10
        words_mat = []
        alpha = []
        from_mat = []

        words_idxs = []


        for t in range(len(words)):
            if words[t] in [' ', '.', ',', '$', '']:
                continue
            words_idxs.append(t)
            if len(alpha) == 0:
                top_n_matches, matched_prob  = self.word_list_dataset.top_n(words[t], n, max_corr_a=max_corr_a, E=E)
                words_mat.append(top_n_matches)
                alpha.append(matched_prob)
                from_mat.append([None for _ in range(n)])
                continue
            top_n_matches, matched_prob  = self.word_list_dataset.top_n(words[t], n, max_corr_a=max_corr_a, E=E)
            words_mat.append(top_n_matches)
            alpha.append([0 for _ in range(n)])
            from_mat.append([0 for _ in range(n)])
            for i in range(n):
                for j in range(n):
                    trans = self.freq_bi[words_mat[-2][j], words_mat[-1][i]]
                    trans = trans if trans > 0 else 0.5
                    if alpha[-1][i] < alpha[-2][j] * trans:
                        alpha[-1][i] = alpha[-2][j] * trans
                        from_mat[-1][i] = j
                alpha[-1][i] *= matched_prob[i]
            alpha[-1] = normalize(alpha[-1])

        
        best_idx_pred = [np.argmax(alpha[-1])]
        for i in range(1, len(from_mat)):
            best_idx_pred.append(from_mat[-i][best_idx_pred[-1]])

        best_idx_pred.reverse()
        best_words_pred = []
        for i, idx_pred in enumerate(best_idx_pred):
            best_words_pred.append(words_mat[i][idx_pred])

        pred_bigram = ''
        for t in range(len(words)):
            if t in words_idxs:
                pred_bigram+=best_words_pred.pop(0)
            else:
                pred_bigram+=words[t]
        return pred_bigram

    def Viterbi_leven(self, text):
        words = re.split('( |\.|,|\$)', text)
        n = 10
        words_mat = []
        alpha = []
        from_mat = []

        words_idxs = []


        for t in range(len(words)):
            if words[t] in [' ', '.', ',', '$', '']:
                continue
            words_idxs.append(t)
            if len(alpha) == 0:
                top_n_matches, matched_prob  = self.word_list_dataset.top_n_leven(words[t], n)
                words_mat.append(top_n_matches)
                alpha.append(matched_prob)
                from_mat.append([None for _ in range(n)])
                continue
            top_n_matches, matched_prob  = self.word_list_dataset.top_n_leven(words[t], n)
            words_mat.append(top_n_matches)
            alpha.append([0 for _ in range(n)])
            from_mat.append([0 for _ in range(n)])
            for i in range(n):
                for j in range(n):
                    trans = self.freq_bi[words_mat[-2][j], words_mat[-1][i]]
                    trans = trans if trans > 0 else 0.5
                    if alpha[-1][i] < alpha[-2][j] * trans:
                        alpha[-1][i] = alpha[-2][j] * trans
                        from_mat[-1][i] = j
                alpha[-1][i] *= matched_prob[i]
            alpha[-1] = normalize(alpha[-1])

        
        best_idx_pred = [np.argmax(alpha[-1])]
        for i in range(1, len(from_mat)):
            best_idx_pred.append(from_mat[-i][best_idx_pred[-1]])

        best_idx_pred.reverse()
        best_words_pred = []
        for i, idx_pred in enumerate(best_idx_pred):
            best_words_pred.append(words_mat[i][idx_pred])

        pred_bigram = ''
        for t in range(len(words)):
            if t in words_idxs:
                pred_bigram+=best_words_pred.pop(0)
            else:
                pred_bigram+=words[t]
        return pred_bigram