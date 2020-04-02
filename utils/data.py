from typing import List

import numpy as np
import torch
import torch.nn as nn


class Data():

    def __init__(self, data: List):
        self._data = data

        self._int2char = {}
        self._char2int = {}

        self._unq_chars = None

        self._input_seq = []
        self._target_seq = []

        self._sentence_maxlen = None
        self._batch_size = None
        self._seq_len = None
        self.dict_size = None

        self._take_chars()
        self._init_params()
        self._create_mapping()
        self._normalize_len()
        print(self._sentence_maxlen)

    def _take_chars(self):
        uniques = set(''.join(self._data))
        self._unq_chars = list(uniques)
        self._unq_chars.sort()

    def _init_params(self):
        self._batch_size = len(self._data)
        self._sentence_maxlen = len(max(self._data, key=len))
        self._seq_len = self._sentence_maxlen - 1
        self.dict_size = len(self._unq_chars)

    def _create_mapping(self):
        for i, char in enumerate(self._unq_chars):
            self._int2char[i] = char
            self._char2int[char] = i

    def _normalize_len(self):
        for i in range(self._batch_size):
            while len(self._data[i]) < self._sentence_maxlen:
                self._data[i] += ' '

    def _create_sequences(self):
        for i in range(self._batch_size):
            self._input_seq.append(self._data[i][:-1])
            self._target_seq.append(self._data[i][1:])

    def _seq2int(self):
        for i in range(self._batch_size):
            self._input_seq[i] = [self._char2int[character] for character in self._input_seq[i]]
            self._target_seq[i] = [self._char2int[character] for character in self._target_seq[i]]

    def _onehot_encode(self, sequence, dict_size, seq_len, batch_size):
        encoded = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
        for i in range(batch_size):
            for u in range(seq_len):
                encoded[i, u, sequence[i][u]] = 1
        return torch.from_numpy(encoded)

    def preprocess(self):
        self._create_sequences()
        self._seq2int()
        return self._onehot_encode(self._input_seq, self.dict_size, self._seq_len, self._batch_size), \
               torch.Tensor(self._target_seq)

    def _predict(self, model, character):
        character = np.array([[self._char2int[c] for c in character]])
        character = self._onehot_encode(character, self.dict_size, character.shape[1], 1)
        out = model(character)
        prob = nn.functional.softmax(out[-1], dim=0).data
        char_ind = torch.max(prob, dim=0)[1].item()
        return self._int2char[char_ind]

    def sample(self,model,start):
        model.eval()
        start = start.lower()
        chars = [ch for ch in start]
        size = self._sentence_maxlen - len(chars)
        for ii in range(size):
            char = self._predict(model, chars)
            chars.append(char)
        return ''.join(chars)