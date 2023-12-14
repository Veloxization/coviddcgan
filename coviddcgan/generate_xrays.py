from keras.models import model_from_json
from keras.utils import array_to_img

import os

import numpy as np

class Generator():
    def __init__(self, model_name):
        f = open(f"saved_model/{model_name}/generator.json")
        generator_data = f.read()
        self.generator = model_from_json(generator_data)
        f.close()
        self.latent_dim = 100
        self.generator.load_weights(f"saved_model/{model_name}/generator_weights.hdf5")

    def generate_images(self, img_amount, save_to_npy=True, img_label=0, save_as_png=False):
        noise = np.random.normal(0, 1, (img_amount, self.latent_dim))
        labels = [img_label for _ in range(img_amount)]
        imgs = self.generator.predict(noise)
        dataset = np.squeeze(imgs, axis=(3,))
        dataset = dataset + 1
        dataset = dataset * 127.5
        dataset = (np.rint(dataset)).astype(np.uint8)
        if save_to_npy:
            np.savez(f"dataset_label_{img_label}.npz", x_train=dataset, y_train=labels)
        if save_as_png:
            for i in range(img_amount):
                img = array_to_img(imgs[i])
                img.save(f"gen_imgs/{i}.png")

if __name__ == "__main__":
    generator = Generator("normalmodel")
    generator.generate_images(11439, save_as_png=True, img_label=1)
