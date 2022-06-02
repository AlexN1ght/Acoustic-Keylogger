import os

from torch.utils.data import Dataset

from .hmm_helpers import *


ALLOWED_CHARS = 'abcdefghijklmnopqrstuvwxyz., '
WORD_KEYS = 'abcdefghijklmnopqrstuvwxyz'

def distance(a, b):
        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n, m)) space
            a, b = b, a
            n, m = m, n

        current_row = range(n + 1)  # Keep current and previous row, not entire matrix
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[n]


class TextDataset(object):
    def __init__(self, dataset_path, context_size=2):
        self.context_size = context_size

        self.shape = tuple([len(ALLOWED_CHARS)])

        text_file = open(dataset_path, encoding='utf8')
        text = text_file.read()
        text_file.close()

        text = text.lower()
        text = text.replace('\n', ' ')

        translation_table = dict.fromkeys(map(ord, list(set(list(text)) - set(list(ALLOWED_CHARS)))), None)
        text = text.translate(translation_table)

        
        text_list = []
        pred_ch = ''
        for ch in text:
            if not (ch in ['.', ',', ' '] and pred_ch == ' '):
                text_list.append(pred_ch)
            pred_ch = ch
        if not (ch in ['.', ','] and pred_ch == ' '):
            text_list.append(pred_ch)
        self.text = ''.join(text_list)

class WordListDataset(Dataset):
    def __init__(self, folder_path, wods_type, sizes, keys=WORD_KEYS):
        path_wo_size = os.path.join(folder_path, wods_type+'.')
        self.data = []

        for size in sizes:
            text_file = open(path_wo_size+size, encoding='ISO-8859-1')
            file_buffer = text_file.read()
            text_file.close()
            self.data += file_buffer.split('\n')

        for i in range(len(self.data)):
            self.data[i] = self.data[i].lower()

        keys_in_wordlist = set()
        for i in range(len(self.data)):
            keys_in_wordlist = keys_in_wordlist.union(set(list(self.data[i])))
        translation_table = dict.fromkeys(map(ord, list(keys_in_wordlist - set(list(keys)))), None)
        
        for i in range(len(self.data)):
            self.data[i] = self.data[i].translate(translation_table)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def merge(self, wordlist):
        self.data += wordlist.data

    def best_match(self, word, E, max_corr_a=1, keys=WORD_KEYS):
        len_word = len(word)
        id_m = id_map(keys)
        best_index = 0
        max_p = -1
        if word in self.data:
            return word
        for i in range(len(self.data)):
            if len(self.data[i]) == len_word:
                corrections = 0
                curr_p = 1
                for j in range(len_word):
                    if self.data[i][j] != word[j]:
                        corrections += 1
                    curr_p *= E[id_m[word[j]], id_m[self.data[i][j]]]
                if corrections / len_word > max_corr_a:
                    continue
                if curr_p > max_p:
                    max_p = curr_p
                    best_index = i
        if max_p == -1:
            return word
        return self.data[best_index]

    def top_n(self, word, n, E, max_corr_a=1, keys=WORD_KEYS):
        len_word = len(word)
        id_m = id_map(keys)
        top_n = []

        for i in range(len(self.data)):
            if len(self.data[i]) == len_word:
                corrections = 0
                curr_p = 1
                for j in range(len_word):
                    if self.data[i][j] != word[j]:
                        corrections += 1
                    curr_p *= E[id_m[word[j]], id_m[self.data[i][j]]]
                if corrections / len_word > max_corr_a:
                    if len_word > 2:
                        continue
                    if corrections == 2:
                        continue
                if len(top_n) != n:
                    top_n.append([self.data[i], curr_p])
                    top_n.sort(key=lambda x:x[1], reverse=True)
                elif curr_p > top_n[-1][1]:
                    top_n.pop()
                    top_n.append([self.data[i], curr_p])
                    top_n.sort(key=lambda x:x[1], reverse=True)

        if len(top_n) == 0:
            top_n = [[word, 1] for _ in range(n)]
        
        while len(top_n) < n:
            top_n.append(['', 0])
        
        matched_words = []
        matched_proba = []
        for i in top_n:
            matched_words.append(i[0])
            matched_proba.append(i[1])
        return matched_words, matched_proba

    def top_n_leven(self, word, n, keys=WORD_KEYS):
        len_word = len(word)
        top_n = []

        for i in range(len(self.data)):
            if abs(len(self.data[i]) - len_word) <= 1:
                curr_p = 1 / (distance(self.data[i], word) + 1)
                if len(top_n) != n:
                    top_n.append([self.data[i], curr_p])
                    top_n.sort(key=lambda x:x[1], reverse=True)
                elif curr_p > top_n[-1][1]:
                    top_n.pop()
                    top_n.append([self.data[i], curr_p])
                    top_n.sort(key=lambda x:x[1], reverse=True)

        if len(top_n) == 0:
            top_n = [[word, 1] for _ in range(n)]
        
        while len(top_n) < n:
            top_n.append(['', 0])
        
        matched_words = []
        matched_proba = []
        for i in top_n:
            matched_words.append(i[0])
            matched_proba.append(i[1])
        return matched_words, matched_proba


    def find_word(self, word):
        return True if word in self.data else False

    def distance(self, a, b):
        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n, m)) space
            a, b = b, a
            n, m = m, n

        current_row = range(n + 1)  # Keep current and previous row, not entire matrix
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[n]
