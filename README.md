# Burst Photography for Learning to Enhance Extremely Dark Images

This is a Tensorflow implementation of Burst Photography for Learning to Enhance Extremely Dark Images, by [Ahmet Serdar Karadeniz](https://askaradeniz.github.io), [Erkut Erdem](https://web.cs.hacettepe.edu.tr/~erkut/), [Aykut Erdem](https://web.cs.hacettepe.edu.tr/~aykut/).


[Project Website](https://hucvl.github.io/dark-burst-photography)

[Paper](https://arxiv.org/pdf/2006.09845.pdf)

[Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Getting Started

### Dependencies

Required python libraries: tensorflow (>=1.1), rawpy, opencv, numpy, scikit-image, scipy, lpips_tf, easydict

For the video model, dependencies of [RAFT](https://github.com/princeton-vl/RAFT)

### Testing

1. Clone this repository.
2. Download the [pretrained models](https://drive.google.com/file/d/1-8VdqvM3K6K2c7LjeNnbiyecWcpdLfIF/view?usp=sharing) and put them to the folder scheckpoint/Sony/burst_l1_cx and checkpoint/Fuji/burst_fuji.
3. Download the [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark).
4. Run `python test.py`

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

1. Clone this repository.
2. Download the [pretrained model](https://drive.google.com/file/d/1-74CghpfYES7QhYXn1N-DhL4TF0U44Oe/view?usp=sharing) and put it to the folder src/seeing-motion/checkpoints/burst_l1_drv_full.
3. Download the [DRV dataset](https://github.com/cchen156/Seeing-Motion-in-the-Dark).
4. Run `python test_image_dbp.py` (static videos) or `python test_video_dbp.py` (dynamic videos) for testing and `python train_dbp.py` for training.

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