# reconstruction-network-for-video-captioning

This project tries to implement *RecNet* proposed on **[Reconstruction Network for Video Captioning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Reconstruction_Network_for_CVPR_2018_paper.pdf)** in **CVPR 2018**.



## Requirements

* Java 1.8
* Python 2.7.12
  * PyTorch 1.0
  * Other python libraries specified in requirements.txt



## How to use

### Setup environment

1. Install Java 8.
2. Setup python virtual environment.
   ```
   $ virtualenv .env
   $ source .env/bin/activate
   $ pip install --upgrade pip
   $ pip install -r requirements.txt
   ```

### Prepare Data

1. Extract feature vectors of datasets by following instructions in *TBD*, and locate them at `~/<dataset>/features/<network>.hdf5`
   
   > e.g. InceptionV4 feature vectors of MSVD dataset will be located at `~/data/MSVD/features/InceptionV4.hdf5`.

2. Set hyperparameters in `config.py` and split the dataset into train / val / test dataset by running following command.
   
   ```
   $ python -m scripts.split
   ```
   

### Train

1. Set hyperparameters in `config.py`.
2. Run
   ```
   $ python train.py
   ```


### Inference

1. Set hyperparameters in `config.py`.
2. Run
   ```
   $ python run.py
   ```


## Result

### Comparison with original paper

* MSVD

  |      | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: |
  | Ours (global) | 40.7 | 27.3 | 34.4 | 61.9 |
  
  | Paper (local) | 52.3 | 34.1 | 69.8 | 80.7 |

* MSR-VTT

  |      | BLEU4 | METEOR | CIDEr | ROUGE_L |
  | :---: | :---: | :---: | :---: | :---: |
  | Ours | - | - | - | - |
  | Paper (local) | 39.1 | 26.6 | 59.3 | 42.7 |


### loss

![image](https://user-images.githubusercontent.com/17702664/49371473-e364d480-f73a-11e8-809b-107ed321e841.png)


### evaluation metrics (BLEU, METEOR, CIDEr, and ROUGE_L)

![image](https://user-images.githubusercontent.com/17702664/49371614-6b4ade80-f73b-11e8-8cec-e0e4dc6b8fb8.png)


## TODO

* Add C3D feature vectors.
* Implement a local reconstructor network.
* Add MSR-VTT dataset.
