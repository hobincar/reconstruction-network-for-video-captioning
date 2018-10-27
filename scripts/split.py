import h5py
import os
import pandas as pd
import random

from config import SplitConfig as C

random.seed(C.random_seed)


def load_metadata():
    df = pd.read_csv(C.caption_fpath)
    df = df[df['Language'] == 'English']
    df = df[pd.notnull(df['Description'])]
    df = df.reset_index(drop=True)
    return df

def load_videos():
    return h5py.File(C.video_fpath, 'r')


def save_video(fpath, vids, videos):
    fout = h5py.File(fpath, 'w')
    for vid in vids:
        fout[vid] = videos[vid].value
    fout.close()
    print("Saved {}".format(fpath))


def save_metadata(fpath, vids, metadata_df):
    vid_indices = [ i for i, r in metadata_df.iterrows() if "{}_{}_{}".format(r[0], r[1], r[2]) in vids ]
    df = metadata_df.iloc[vid_indices]
    df.to_csv(fpath)
    print("Saved {}".format(fpath))

def split():
    video_dpath = os.path.dirname(C.video_fpath)
    caption_dpath = os.path.dirname(C.caption_fpath)

    videos = load_videos()
    metadata = load_metadata()

    vids = list(videos.keys())
    random.shuffle(vids)

    train_vids = vids[:C.n_train]
    val_vids = vids[C.n_train:C.n_train+C.n_val]
    test_vids = vids[C.n_train+C.n_val:]

    save_video(C.train_video_fpath, train_vids, videos)
    save_video(C.val_video_fpath, val_vids, videos)
    save_video(C.test_video_fpath, test_vids, videos)
    
    save_metadata(C.train_metadata_fpath, train_vids, metadata)
    save_metadata(C.val_metadata_fpath, val_vids, metadata)
    save_metadata(C.test_metadata_fpath, test_vids, metadata)


if __name__ == "__main__":
    split()

