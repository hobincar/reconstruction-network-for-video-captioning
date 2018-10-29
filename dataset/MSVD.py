from __future__ import print_function, division

from collections import defaultdict
import os

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.transform import UniformSample, ZeroPadIfLessThan, ToTensor, RemovePunctuation, Lowercase, \
                      SplitWithWhiteSpace, Truncate, ToIndex


class MSVD:
    """ MSVD DataLoader """
    
    def __init__(self, C):
        self.C = C
        self.vocab = None
        self.data_loader = None

        self.transform_sentence = transforms.Compose([
            RemovePunctuation(),
            Lowercase(),
            SplitWithWhiteSpace(),
            Truncate(self.C.caption_n_max_word),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        self.build_data_loader()

    def build_vocab(self):
        self.vocab = MSVDVocab(
            self.C.total_caption_fpath,
            self.C.init_word2idx,
            self.C.min_count,
            transform=self.transform_sentence)

    def build_data_loader(self):
        def collate_fn(batch):
            n_sample = len(batch)
            frames = [ frame for frame, _ in batch ]
            captions = [ caption.long() for _, caption in batch ]

            # For a caption shorter than the longest one, pad with zeros
            max_n_words = max([ len(caption) for caption in captions ])
            new_captions = []
            masks = []
            for caption in captions:
                n_words = len(caption)
                n_pads = max_n_words - n_words
                new_caption = torch.cat(( caption, torch.zeros(n_pads, dtype=torch.long) ))
                mask = torch.cat(( torch.ones(n_words), torch.zeros(n_pads) )).byte()
                new_captions.append(new_caption)
                masks.append(mask)

            batch_frames = torch.stack(frames, dim=0)
            batch_captions = torch.stack(new_captions, dim=0).transpose(0, 1)
            batch_masks = torch.stack(masks, dim=0).transpose(0, 1)

            return batch_frames, batch_captions, batch_masks, max_n_words

        transform_frame=transforms.Compose([
            UniformSample(self.C.encoder_output_len),
            ZeroPadIfLessThan(self.C.encoder_output_len),
            ToTensor(),
        ])
        transform_caption=transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            ToTensor(),
        ])

        train_dataset = MSVDDataset(
            self.C.train_video_fpath,
            self.C.train_caption_fpath,
            transform_frame=transform_frame,
            transform_caption=transform_caption)

        self.train_data_loader = DataLoader(
            train_dataset,
            batch_size=self.C.batch_size,
            shuffle=self.C.shuffle,
            num_workers=self.C.num_workers,
            collate_fn=collate_fn,
        )

        val_dataset = MSVDDataset(
            self.C.val_video_fpath,
            self.C.val_caption_fpath,
            transform_frame=transform_frame,
            transform_caption=transform_caption)

        self.val_data_loader = DataLoader(
            val_dataset,
            batch_size=self.C.batch_size,
            shuffle=self.C.shuffle,
            num_workers=self.C.num_workers,
            collate_fn=collate_fn,
        )


class MSVDVocab:
    """ MSVD Vocaburary """

    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=None):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform

        self.word2idx = init_word2idx
        self.idx2word = { v: k for k, v in self.word2idx.items() }
        self.word_freq_dict = defaultdict(lambda: 0)
        self.n_vocabs = len(self.word2idx)
        self.n_words = self.n_vocabs

        self.build()

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[pd.notnull(df['Description'])]
        captions = df['Description'].values
        return captions

    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption) if self.transform else caption.split()
            for word in words:
                self.word_freq_dict[word] += 1
        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        keep_words = [ word for word, freq in self.word_freq_dict.items() if freq >= self.min_count ]

        for idx, word in enumerate(keep_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.n_vocabs = len(keep_words)
        self.n_words = sum([ self.word_freq_dict[word] for word in keep_words ])


class MSVDDataset(Dataset):
    """ MSVD Dataset """

    def __init__(self, video_fpath, caption_fpath, transform_frame=None, transform_caption=None):
        self.video_fpath = video_fpath
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption

        self.build_video_caption_pairs()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, caption = self.video_caption_pairs[idx]

        if self.transform_frame:
            video = self.transform_frame(video)
        if self.transform_caption:
            caption = self.transform_caption(caption)

        return video, caption
    
    def load_videos(self):
        fin = h5py.File(self.video_fpath, 'r')
        videos = {}
        for vid in fin:
            videos[vid] = fin[vid].value
        self.videos = videos
        return videos

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
        df = df[pd.notnull(df['Description'])]

        captions = defaultdict(lambda: [])
        for video_id, start, end, caption in df.values:
            vid = "{}_{}_{}".format(video_id, start, end)
            captions[vid].append(caption)
        self.captions = captions
        return captions

    def build_video_caption_pairs(self):
        self.load_videos()
        self.load_captions()

        self.video_caption_pairs = []
        for vid in self.videos:
            video = self.videos[vid]
            for caption in self.captions[vid]:
                self.video_caption_pairs.append(( video, caption ))
        return self.video_caption_pairs

