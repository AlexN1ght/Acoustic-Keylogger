import re

import matplotlib.pyplot as plt
from colorama import Fore

import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from .hmm_helpers import *
from .keySoundDataset import *
from .textDataset import *
from .clustering import *


def create_extended_transmat(text):
    trans_mat_int, keys = create_transmat(text)
    err_percent = 0.005
    err_after_err_percent = 0.08
    trans_mat = np.zeros((len(keys) + 1, len(keys) + 1), dtype=np.float64)
    for i, line in enumerate(trans_mat_int):
        if np.sum(line) <= 0:
            line = np.ones((len(keys)), dtype=np.float64)

        line_w_err_sum = np.sum(line)
        line = np.array(list(line) + [(line_w_err_sum / (1-err_percent)) * err_percent])

        trans_mat[i] = line / np.sum(line)

    line = np.ones((len(keys)), dtype=np.float64)
    line = np.array(list(line) + [(np.sum(line) / (1-err_after_err_percent)) * err_after_err_percent])
    trans_mat[-1] = line / np.sum(line)
    keys = keys + '$'
    return trans_mat, keys

def pprint_pred(original_text, pred_text):
    for i in range(len(pred_text)):
        if pred_text[i] == original_text[i]:
            print(Fore.BLUE + pred_text[i], end='')
        else:
            print(Fore.RED + pred_text[i], end='')


    
def basic_spell_check(text, word_list_dataset, E, max_corr_a=1):
    pred_text = ''

    words = re.split('( |\.|,|\$)', text)
    for word in words:
        if word in [',', '.', ' ', '$', '']:
            pred_text += word
            continue
        word = word_list_dataset.best_match(word, E, max_corr_a=max_corr_a, keys=WORD_KEYS)
        pred_text += word
    return pred_text

def basic_spell_check_leven(text, word_list_dataset):
    pred_text = ''

    words = re.split('( |\.|,|\$)', text)
    for word in words:
        if word in [',', '.', ' ', '$', '']:
            pred_text += word
            continue
        word = word_list_dataset.top_n_leven(word, 1)[0][0]
        pred_text += word
    return pred_text


def show_keys_confusion_matrix(pred_text_encoded, true_lbls, keys):
    lbls_for_cm = []

    id_m = id_map(keys)
    for ch in true_lbls:
        if ch not in id_m:
            lbls_for_cm.append(id_m['$'])
        else:
            lbls_for_cm.append(id_m[ch])

    cm = confusion_matrix(lbls_for_cm, pred_text_encoded, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=list(keys))
    _, ax = plt.subplots(figsize=(20,20))
    disp.plot(ax=ax)
    plt.show()


def word_precision(text, pred_text):
    alphabet = WORD_KEYS
    word_status = 0
    cnt = 0
    true_cnt = 0
    for i in range(len(text)):
        cur_true_ch = text[i]
        if cur_true_ch in alphabet:
            if cur_true_ch == pred_text[i]:
                if word_status == 0:
                    word_status = 1
            else:
                word_status = -1
        else:
            if word_status != 0:
                cnt += 1
            if word_status == 1:
                true_cnt += 1
            word_status = 0
    if word_status != 0:
        cnt += 1
    if word_status == 1:
        true_cnt += 1

    return true_cnt/cnt


def get_cond_prob_matrix(textDataset):
    char_frq = [[i ,textDataset.text.count(i)/len(textDataset.text)] for i in WORD_KEYS]
    char_frq.sort(key=lambda x: x[1])

    def f(x):
        return (-0.96*pow(x, 1.5) + 8*x + 10) / 100

    for i in range(len(WORD_KEYS)):
        char_frq[i][1]=f(i)

    char_frq.sort(key=lambda x: x[0])
    char_frq = [i[1] for i in char_frq]

    E = np.zeros((len(WORD_KEYS), len(WORD_KEYS)), dtype=np.float32)

    for i in range(len(WORD_KEYS)):
        sum_over_rest = sum(char_frq) - char_frq[i]
        for j in range(len(WORD_KEYS)):
            if i == j:
                E[i, i] = char_frq[i]
                continue
            E[i, j] = (char_frq[j] / sum_over_rest) * (1 - char_frq[i])
    return E

def get_unified_cond_prob_matrix(confidence=0.5):
    E = np.zeros((len(WORD_KEYS), len(WORD_KEYS)), dtype=np.float32)

    for i in range(len(WORD_KEYS)):
        E[i] = np.ones((len(WORD_KEYS)), dtype=np.float32) * (1 - confidence) / (len(WORD_KEYS) - 1)
        E[i][i] = confidence
    return E

def format_spell_check(pred_text, original_text,  word_list_dataset, min_length=0, verbose=True):
    cnt = 0
    true_cnt = 0
    i = 0
    matched_indexes = []
    matched_str = ''
    words = re.split('( |\.|,|\$)', pred_text)
    for word in words:
        if word == '':
            continue
        if word in [',', '.', ' ', '$']:
            matched_str += word
            matched_indexes.append(i)
            if word[0] == original_text[i]:
                if verbose:
                    print(Fore.GREEN + word[0], end='')
                true_cnt += 1
            else:
                if verbose:
                    print(Fore.RED + word[0], end='')
            cnt += 1
        elif min_length <= len(word) and word_list_dataset.find_word(word):
            for j in range(len(word)):
                matched_str += word[j]
                matched_indexes.append(i + j)
                if word[j] == original_text[i + j]:
                    if verbose:
                        print(Fore.GREEN + word[j], end='')
                    true_cnt += 1
                else:
                    if verbose:
                        print(Fore.RED + word[j], end='')
            cnt += len(word)
        else:
            if verbose:
                print(Fore.BLUE + word, end='')
        i += len(word)

    return true_cnt / cnt, matched_str, matched_indexes

def words_in_wordlist(pred_text,  word_list_dataset, min_length=0):
    matched_str = ''
    words = re.split('( |\.|,|\$)', pred_text)
    for word in words:
        if word in [',', '.', ' ', '$', '']:
            continue
        if min_length <= len(word) and word_list_dataset.find_word(word):
            matched_str += word

    return matched_str


