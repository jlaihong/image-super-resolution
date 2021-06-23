# Image Super-Resolution using SRResNet and SRGAN

A TensorFlow 2 implementation of the paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802.pdf)


## Training

Follow the code in SRRestNet_and_SRGAN_train.ipynb

The models train using the [div2k dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

You can select from the different div2k datasets by changing the key:

```python
dataset_key = "bicubic_x4" # by default
```

dataset_key = "bicubic_x4"

# SRResNet:

 


## Acknowledgements

Much of the code in this repository has been refactored from 
https://github.com/krasserm/super-resolution