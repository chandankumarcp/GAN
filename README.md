# GAN on CIFAR-10

A DCGAN implemented with Keras/TensorFlow that generates 32x32 CIFAR-10-style images.

## Structure

| File | Purpose |
|---|---|
| `explore_data.py` | Plot 49 sample CIFAR-10 training images |
| `data.py` | Load CIFAR-10 and scale to [-1, 1] |
| `models.py` | Discriminator, generator and combined GAN |
| `train.py` | Training loop; saves generator to `cifar10_generator.h5` |
| `generate.py` | Load the saved generator and plot generated images |

## Usage

```bash
pip install -r requirements.txt
python explore_data.py   # optional
python train.py          # trains and saves cifar10_generator.h5
python generate.py       # shows 16 generated images
```

The colab link-https://colab.research.google.com/drive/1-mWnnVe1WGNqpje89sStfkQ1jsYBKK2k?usp=sharing

The reference link-https://machinelearningmastery.com/generative-adversarial-network-loss-functions/
