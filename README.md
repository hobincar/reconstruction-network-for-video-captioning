# reconstruction-network-for-video-captioning

This project tries to implement *RecNet* proposed on **[Reconstruction Network for Video Captioning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf)** in **CVPR 2018**.



## Requirements

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Java 1.8
* Python 2.7.12
  * PyTorch 1.0
  * Other python libraries specified in requirements.txt



## How to use

### Step 1. Setup python virtual environment

```
$ pip install virtualenv
$ virtualenv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```


### Step 2. Prepare Data

1. Extract feature vectors of datasets by following instructions in [here](https://github.com/hobincar/awesome-video-dataset), and locate them at `~/<dataset>/features/<network>.hdf5`
   
   > e.g. InceptionV4 feature vectors of MSVD dataset will be located at `~/data/MSVD/features/InceptionV4.hdf5`.

2. Set hyperparameters in `config.py` and split the dataset into train / val / test dataset by running following command.
   
   ```
   (.env) $ python -m scripts.split
   ```
   

### Step 3. Train

1. Set hyperparameters in `config.py`.
2. Run
   ```
   (.env) $ python train.py
   ```


### Step 4. Inference

1. Set hyperparameters in `config.py`.
2. Run
   ```
   (.env) $ python run.py
   ```


## Result

### Comparison with original paper

**NOTE**: *For now, only 2D features are used for evaluating our model (3D features are missing).*

* MSVD

  |   | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: |
  | Ours (wo. reconstructor) | 39.4 | 27.2 | 37.8 | 61.8 |
  | Ours (global) | 40.7 | 27.3 | 34.4 | 61.9 |
  | Ours (local) | 35.3 | 27.3 | 35.2 | 61.9 |
  | Paper (global) | 51.1 | 34.0 | 69.4 | 79.7 |
  | **Paper (local)** | **52.3** | **34.1** | **69.8** | **80.7** |

<!--
[Ours (wo. reconstructor)] (100k) RecNet | MSVD tc-30 mc-5 sp-uniform | ENC InceptionV4 sm-28 | DEC LSTM-1 at-128 dr-0.5-0.5 tf-1.0 lr-1e-05-wd-1e-05 op-amsgrad | EMB 468 dr-0.5 sc-1 | bs-100 | cp-50.0 | 181204-15:25:22

[Ours (global)]: (100k) RecNet | MSVD tc-30 mc-5 sp-uniform | ENC InceptionV4 sm-28 | DEC lstm-1 at-128 dr-0.5-0.5 tf-1.0 lr-1e-05-wd-1e-05 op-amsgrad | REC LSTM lr-1e-06-wd-1e-05 op-adam | EMB 468 dr-0.5 sc-1 | bs-100 | cp-50.0 | 181117-18:32:22

[Ours (local)] (100k) RecNet | MSVD tc-30 mc-5 sp-uniform | ENC InceptionV4 sm-28 | DEC LSTM-1 at-128 dr-0.5-0.5 tf-1.0 lr-1e-05-wd-1e-05 op-amsgrad | REC-local LSTM lr-1e-06-wd-1e-05 op-adam at-128 | EMB 468 dr-0.5 sc-1 | bs-100 | cp-50.0 | 181204-15:26:31
-->

* MSR-VTT

  |   | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: |
  | Ours | - | - | - | - |
  | Paper (global) | 38.3 | 26.2 | 59.1 | 41.7 |
  | **Paper (local)** | **39.1** | **26.6** | **59.3** | **42.7** |


## TODO

* Add qualitative results
* Add C3D feature vectors.
* Add MSR-VTT dataset.
