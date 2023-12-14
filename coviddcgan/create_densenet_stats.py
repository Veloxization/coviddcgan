import matplotlib.pyplot as plt
import numpy as np

from keras.models import model_from_json

plt.style.use('_mpl-gallery')

densenet_normal = None
densenet_augmented = None
epochs = 100
batch_size = 2000
# Prepare models
with open(f"saved_densenet/real_dataset/densenet.json", encoding="UTF-8") as f:
    densenet_data = f.read()
    densenet_normal = model_from_json(densenet_data)

with open(f"saved_densenet/augmented_dataset/densenet.json", encoding="UTF-8") as f:
    densenet_data = f.read()
    densenet_augmented = model_from_json(densenet_data)

# Compile models
densenet_normal.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
densenet_augmented.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

# Load model weights
densenet_normal.load_weights("saved_densenet/real_dataset/densenet_weights.hdf5")
densenet_augmented.load_weights("saved_densenet/augmented_dataset/densenet_weights.hdf5")

# Open dataset containing test data
dataset = np.load("real_dataset.npz")
x_test = dataset["x_test"]
y_test = dataset["y_test"]

# Normalize pixel values to be between 0 and 1
x_test = x_test.astype('float32') / 255.0

# Expand dimensions for grayscale images
x_test = x_test[:, :, :, np.newaxis]
x_test = np.concatenate([x_test, x_test, x_test], axis=-1)

# Prepare data
losses_normal = []
losses_augmented = []
accuracies_normal = []
accuracies_augmented = []
mses_normal = []
mses_augmented = []

for i in range(epochs):
    print(f"epoch {i+1}/{epochs}")

    # Select random images
    idx = np.random.randint(0, x_test.shape[0], batch_size)
    imgs = x_test[idx]
    labels = y_test[idx]

    test_result = densenet_normal.evaluate(imgs, labels, verbose=2)
    losses_normal.append(test_result[0])
    accuracies_normal.append(test_result[1])
    mses_normal.append(test_result[2])

    test_result = densenet_augmented.evaluate(imgs, labels, verbose=2)
    losses_augmented.append(test_result[0])
    accuracies_augmented.append(test_result[1])
    mses_augmented.append(test_result[2])

# Convert data to NumPy arrays
losses_normal = np.asarray(losses_normal)
losses_augmented = np.asarray(losses_augmented)
accuracies_normal = np.asarray(accuracies_normal)
accuracies_augmented = np.asarray(accuracies_augmented)
mses_normal = np.asarray(mses_normal)
mses_augmented = np.asarray(mses_augmented)

print("Stats for non-augmented")
print(f"loss - min: {np.min(losses_normal)} | max: {np.max(losses_normal)} | mean: {np.mean(losses_normal)} | median: {np.median(losses_normal)}")
print(f"accuracy - min: {np.min(accuracies_normal)} | max: {np.max(accuracies_normal)} | mean: {np.mean(accuracies_normal)} | median: {np.median(accuracies_normal)}")
print(f"MSE - min: {np.min(mses_normal)} | max: {np.max(mses_normal)} | mean: {np.mean(mses_normal)} | median: {np.median(mses_normal)}")

print("Stats for augmented")
print(f"loss - min: {np.min(losses_augmented)} | max: {np.max(losses_augmented)} | mean: {np.mean(losses_augmented)} | median: {np.median(losses_augmented)}")
print(f"accuracy - min: {np.min(accuracies_augmented)} | max: {np.max(accuracies_augmented)} | mean: {np.mean(accuracies_augmented)} | median: {np.median(accuracies_augmented)}")
print(f"MSE - min: {np.min(mses_augmented)} | max: {np.max(mses_augmented)} | mean: {np.mean(mses_augmented)} | median: {np.median(mses_augmented)}")

losses = (losses_normal, losses_augmented)
accs = (accuracies_normal, accuracies_augmented)
mses = (mses_normal, mses_augmented)

# plot
fig, axs = plt.subplots(3)
axs[0].boxplot(losses, positions=[1, 2], widths=0.9, patch_artist=True,
               showmeans=True, showfliers=False,
               medianprops={"color": "white", "linewidth": 0.5},
               boxprops={"facecolor": "C0", "edgecolor": "white",
                         "linewidth": 0.5},
               whiskerprops={"color": "C0", "linewidth": 1.5},
               capprops={"color": "C0", "linewidth": 1.5},
               labels=("loss normal", "loss augmented"))

axs[0].set(xlim=(0, 3), xticks=np.arange(1, 3),
           ylim=(1.80, 2.0), yticks=np.arange(1.80, 2.01, 0.02))

axs[1].boxplot(accs, positions=[1, 2], widths=0.9, patch_artist=True,
               showmeans=True, showfliers=False,
               medianprops={"color": "white", "linewidth": 0.5},
               boxprops={"facecolor": "C0", "edgecolor": "white",
                         "linewidth": 0.5},
               whiskerprops={"color": "C0", "linewidth": 1.5},
               capprops={"color": "C0", "linewidth": 1.5},
               labels=("accuracy normal", "accuracy augmented"))

axs[1].set(xlim=(0, 3), xticks=np.arange(1, 3),
           ylim=(0.92, 1.0), yticks=np.arange(0.92, 1.01, 0.01))

axs[2].boxplot(mses, positions=[1, 2], widths=0.9, patch_artist=True,
               showmeans=True, showfliers=False,
               medianprops={"color": "white", "linewidth": 0.5},
               boxprops={"facecolor": "C0", "edgecolor": "white",
                         "linewidth": 0.5},
               whiskerprops={"color": "C0", "linewidth": 1.5},
               capprops={"color": "C0", "linewidth": 1.5},
               labels=("MSE normal", "MSE augmented"))

axs[2].set(xlim=(0, 3), xticks=np.arange(1, 3),
           ylim=(0.44, 0.5), yticks=np.arange(0.44, 0.51, 0.01))

plt.show()
