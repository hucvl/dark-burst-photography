# Burst Photography for Learning to Enhance Extremely Dark Images

This is a Tensorflow implementation of Burst Photography for Learning to Enhance Extremely Dark Images, by [Ahmet Serdar Karadeniz](https://askaradeniz.github.io), [Erkut Erdem](https://web.cs.hacettepe.edu.tr/~erkut/), [Aykut Erdem](https://web.cs.hacettepe.edu.tr/~aykut/).


[Project Website](https://hucvl.github.io/dark-burst-photography)

[Paper](https://arxiv.org/pdf/2006.09845.pdf)

[Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Getting Started

### Dependencies

Required python libraries: tensorflow (>=1.1), rawpy, opencv, numpy, scikit-image, scipy, lpips_tf, easydict

### Testing

1. Clone this repository.
2. Download the [pretrained model](https://drive.google.com/file/d/1u-FG05HBb2h9ws4Xx9TQw272s64emtJR/view?usp=sharing) and put it to the folder checkpoint/Sony/burst_l1_cx.
3. Download the [SID dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark).
4. Run `python3 test.py`

### Training

1. For the perceptual and contextual losses, download the pre-trained VGG-19 model:
    ```
    python3 download_vgg_models.py
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
    n_burst = 10
    ```

2. Train the model
    ```
    python3 train.py
    ```

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