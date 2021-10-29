# Burst Photography for Learning to Enhance Extremely Dark Images

This is a Tensorflow implementation of Burst Photography for Learning to Enhance Extremely Dark Images, by [Ahmet Serdar Karadeniz](https://askaradeniz.github.io), [Erkut Erdem](https://web.cs.hacettepe.edu.tr/~erkut/), [Aykut Erdem](https://web.cs.hacettepe.edu.tr/~aykut/).


[Project Website](https://hucvl.github.io/dark-burst-photography)

[Paper](https://arxiv.org/pdf/2006.09845.pdf)

[Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Getting Started

### Dependencies

Prepare the environment (optional).
```
conda create -n dark-burst-photography python=3.6
conda activate dark-burst-photography
``` 


Clone this repository and install the required libraries.
```
git clone https://github.com/hucvl/dark-burst-photography
cd src
pip install -r requirements.txt

git clone https://github.com/alexlee-gk/lpips-tensorflow
cd lpips-tensorflow
python setup.py install
```

### Testing

1. Download the [pretrained models](https://drive.google.com/file/d/1-8VdqvM3K6K2c7LjeNnbiyecWcpdLfIF/view?usp=sharing) and put them to the folders `checkpoint/Sony/burst_l1_res_se_motion_cx` and `checkpoint/Fuji/burst_fuji`.
2. Download the [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark) or just use the sample images in this repository.
3. Run `python test.py`

### Training

1. For the perceptual and contextual losses, download the pre-trained VGG-19 model:
    ```
    python download_vgg_models.py
    ```

2. For multiscale training, set the following variables inside train.py:


    **Coarse network**
    ```python
    train_coarse = True
    finetune = False
    ```

    **Fine network**
    ```python
    train_coarse = False
    finetune = False
    ```

    **Set-based burst network**
    ```python
    train_coarse = False
    finetune = True
    n_burst = 8
    ```

2. Train the model
    ```
    python train.py
    ```

### Training/testing Video Model

1. Download the [pretrained model](https://drive.google.com/file/d/1-74CghpfYES7QhYXn1N-DhL4TF0U44Oe/view?usp=sharing) and put it to the folder src/seeing-motion/checkpoints/burst_l1_drv_full.
2. Download the [DRV dataset](https://github.com/cchen156/Seeing-Motion-in-the-Dark).
3. Run `python test_image_dbp.py` (static videos) or `python test_video_dbp.py` (dynamic videos) for testing and `python train_dbp.py` for training.

## License
MIT License.


## Citation
If you use this code for your research, please consider citing our paper: 
```
@ARTICLE{dark-burst-photography,
    author={Ahmet Serdar Karadeniz and Erkut Erdem and Aykut Erdem},
    journal={submitted},
    title={Burst Photography for Learning to Enhance Extremely Dark Images},
    year={2020},
    volume={},
    number={},
    pages={1-13},
    month={}
}
```
