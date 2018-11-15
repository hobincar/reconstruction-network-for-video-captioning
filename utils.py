import numpy as np


# Why custom?: https://stackoverflow.com/questions/47714643/pytorch-data-loader-multiple-iterations#answer-49987606
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def convert_idxs_to_sentences(idxs, idx2word, EOS_token):
    sentences = []
    idxs_list = np.asarray(idxs).T
    for idxs in idxs_list:
        sentences.append([])
        for idx in idxs:
            if idx == EOS_token: break
            sentences[-1].append(idx2word[idx.item()])
        sentences[-1] = " ".join(sentences[-1])
    return sentences


def sample_n(lst, n):
    indices = list(range(len(lst)))
    sample_indices = np.random.choice(indices, n, replace=False)
    sample_lst = [ lst[i] for i in sample_indices ]
    return sample_lst

