"""GAN model definitions."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Conv2D, Flatten, Dropout,
                                     LeakyReLU, Reshape, Conv2DTranspose)
from tensorflow.keras.optimizers import Adam


def build_discriminator(input_shape=(32, 32, 3)):
    model = Sequential(name="Discriminator")

    # Block 1
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    # Block 2 (Downsample to 16x16)
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Block 3 (Downsample to 8x8)
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Block 4 (Downsample to 4x4)
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def build_generator(latent_dim):
    model = Sequential(name="Generator")

    # Foundation for 4x4 image
    nodes = 256 * 4 * 4
    model.add(Dense(nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))

    # Upsample to 8x8
    model.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample to 16x16
    model.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Upsample to 32x32
    model.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Output layer
    model.add(Conv2D(3, kernel_size=(3, 3), activation='tanh', padding='same'))
    return model


def build_gan(g_model, d_model):
    # Make discriminator untrainable during generator training
    d_model.trainable = False

    model = Sequential(name="GAN")
    model.add(g_model)
    model.add(d_model)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
