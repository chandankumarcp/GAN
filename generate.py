"""Generate and display images with the trained generator."""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def generate_and_show_images(model_path, latent_dim, num_images=16):
    model = load_model(model_path, compile=False)

    noise = np.random.randn(num_images, latent_dim)
    generated_images = model.predict(noise)

    # Rescale images from [-1, 1] back to [0, 1] for matplotlib
    generated_images = (generated_images + 1) / 2.0

    grid_size = int(np.sqrt(num_images))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            axes[i, j].imshow(generated_images[count])
            axes[i, j].axis('off')
            count += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    generate_and_show_images('cifar10_generator.h5', latent_dim=100, num_images=16)
