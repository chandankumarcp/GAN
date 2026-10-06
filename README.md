# 🎨 GAN on CIFAR-10

A **DCGAN** built with Keras/TensorFlow that learns to generate 32×32 CIFAR-10-style images.

---

## 📖 Description

This repo is a compact, end-to-end implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) in Python. Two networks compete and improve together:

- 🎭 **Generator**: turns random noise into fake images.
- 🕵️ **Discriminator**: decides whether an image is real or fake.

---

## 📁 Project Structure

| File | Purpose |
|---|---|
| `explore_data.py` | Plot 49 sample CIFAR-10 training images |
| `data.py` | Load CIFAR-10 and scale to [-1, 1] |
| `models.py` | Discriminator, generator and combined GAN |
| `train.py` | Training loop; saves generator to `cifar10_generator.h5` |
| `generate.py` | Load the saved generator and plot generated images |
| `requirements.txt` | Python dependencies (TensorFlow, NumPy, Matplotlib) |

---

## 🚀 Usage

```bash
pip install -r requirements.txt
python explore_data.py   # optional
python train.py          # trains and saves cifar10_generator.h5
python generate.py       # shows 16 generated images
```

---

## 🧠 Types of GANs

### 1. Vanilla GAN
The simplest GAN.
- Generator and discriminator are both simple multi-layer perceptrons (MLPs).
- Trained with stochastic gradient descent (SGD).
- ⚠️ Can be unstable to train and give limited variety.
- 💡 *Example:* generating handwritten digits like MNIST.

### 2. Conditional GAN (CGAN)
Adds a condition (label `y`) to guide what gets generated.
- The label is fed into **both** the generator and the discriminator.
- The generator creates data that matches the given label.
- The discriminator uses the label to judge real vs. fake.
- 💡 *Example:* ask for a "dog" or a "cat" instead of a random image.

### 3. Deep Convolutional GAN (DCGAN)
One of the most popular GANs for image generation (the type used in this repo).
- Uses CNNs instead of MLPs.
- Replaces max pooling with strided convolutions for efficiency.
- Removes fully connected layers for better spatial understanding.
- 💡 *Example:* realistic faces or objects from random noise.

### 4. StyleGAN3 (Alias-Free GAN)
Developed by NVIDIA; controls fine "styles" (freckles, hair colour, lighting) at different layers. StyleGAN3 fixes StyleGAN2's **"texture sticking"** problem.
- **How it works:** older versions kept features like beards or wrinkles stuck to screen coordinates, even when a face rotated. StyleGAN3 redesigns the signal processing to be alias-free, so textures move naturally with the geometry.
- ✨ **Impact:** flawless high-resolution portraits and smooth cinematic animations.

### 5. WGAN-GP (Wasserstein GAN with Gradient Penalty)
Fixes the maths problems of standard GANs, where metrics like Jensen-Shannon divergence cause vanishing gradients once the discriminator gets too strong.
- **How it works:** uses the Wasserstein (Earth Mover's) Distance to measure how close generated data is to real data. Instead of clipping weights (original WGAN), it penalizes gradients that stray from a target norm (the gradient penalty).
- ✨ **Impact:** greatly reduces mode collapse and lets the discriminator train fully without breaking the generator.

### 6. CycleGAN
Translates images **without paired data** (no need for a photo matched exactly with its sketch).
- **How it works:** two generators and two discriminators, plus a **Cycle-Consistency Loss**: translate A → B → A and you should get the original image back.
- ✨ **Impact:** unpaired image-to-image translation, e.g. summer → winter, or photos → Monet-style paintings.

### 📊 Quick Comparison

| GAN | Key idea | Best for |
|---|---|---|
| Vanilla GAN | MLP generator & discriminator | Learning the basics |
| CGAN | Conditioned on labels | Controlled generation |
| DCGAN | Convolutional layers | Image generation |
| StyleGAN3 | Alias-free, style control | High-res faces & animation |
| WGAN-GP | Wasserstein distance + gradient penalty | Stable training |
| CycleGAN | Cycle-consistency loss | Unpaired image translation |

---

## 🔗 Links

- 📓 [Colab notebook](https://colab.research.google.com/drive/1-mWnnVe1WGNqpje89sStfkQ1jsYBKK2k?usp=sharing)
- 📚 [Reference: GAN loss functions](https://machinelearningmastery.com/generative-adversarial-network-loss-functions/)
