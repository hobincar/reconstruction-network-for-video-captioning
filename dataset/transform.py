import re
import string

import numpy as np
import torch


class UniformSample:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, frames):
        n_frames = len(frames)
        if n_frames < self.n_sample:
            return frames
        
        sample_indices = [ int(i) for i in np.linspace(0, n_frames-1, self.n_sample) ]
        samples = [ frames[i] for i in sample_indices ]
        return samples


class ZeroPadIfLessThan:
    def __init__(self, n):
        self.n = n

    def __call__(self, frames):
        while len(frames) < self.n:
            frames.append(np.zeros_like(frames[0]))
        return frames


class ToTensor:
    
    def __call__(self, array):
        np_array = np.asarray(array)
        return torch.from_numpy(np_array)


class RemovePunctuation:
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def __call__(self, sentence):
        return self.regex.sub('', sentence)


class Lowercase:

    def __call__(self, sentence):
        return sentence.lower()


class SplitWithWhiteSpace:

    def __call__(self, sentence):
        return sentence.split()


class Truncate:
    def __init__(self, n_word):
        self.n_word = n_word

    def __call__(self, words):
        return words[:self.n_word]


class ToIndex:
    def __init__(self, word2idx):
        self.word2idx = word2idx

    def __call__(self, words): # Ignore unknown (or trimmed) words.
        return [ self.word2idx[word] for word in words if word in self.word2idx ]

