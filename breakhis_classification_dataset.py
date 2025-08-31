import tensorflow as tf
from keras.applications import InceptionV3, EfficientNetB0, ResNet101V2
from keras import layers, models, Input, Model
from keras.layers import Average
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import collections
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#DATASET preperation
AUTOTUNE = tf.data.AUTOTUNE

def get_class_distribution(dataset):
    class_counts = collections.Counter()

    for images, labels in dataset:
        labels = labels.numpy()
        class_indices = np.argmax(labels, axis=1)  # convert one-hot to class index
        class_counts.update(class_indices.tolist())

    return dict(sorted(class_counts.items()))


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)

dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode="categorical"
)
class_names = dataset.class_names

total_batches = tf.data.experimental.cardinality(dataset).numpy()
train_batches = int(total_batches * 0.75)  # 75% of 80% = 60%
val_batches = total_batches - train_batches  # 25% of 80% = 20%

dataset_train = dataset.take(train_batches)
dataset_validation = dataset.skip(train_batches)

dataset_train = dataset_train.cache().prefetch(buffer_size=AUTOTUNE)
dataset_validation = dataset_validation.cache().prefetch(buffer_size=AUTOTUNE)
dataset_test = dataset_test.cache().prefetch(buffer_size=AUTOTUNE)

train_dist = get_class_distribution(dataset_train)
validation_dist = get_class_distribution(dataset_validation)
test_dist = get_class_distribution(dataset_test)

print("Train class distribution:", train_dist)
print("Validation class distribution:", validation_dist)
print("Test class distribution:", test_dist)

# Pretvori class_names u indeksiranu mapu
class_to_image = {}
class_names = dataset.class_names

for images, labels in dataset.unbatch():
    class_index = np.argmax(labels.numpy())
    if class_index not in class_to_image:
        class_to_image[class_index] = images.numpy()
    
    if len(class_to_image) == len(class_names):
        break

# Prikaz po jedne slike za svaku klasu
plt.figure(figsize=(15, 8))
for idx, (class_index, image) in enumerate(sorted(class_to_image.items())):
    plt.subplot(2, (len(class_names) + 1) // 2, idx + 1)
    plt.imshow(image.astype(np.uint8))
    plt.title(f"{class_names[class_index]}")
    plt.axis("off")

plt.tight_layout()
plt.show()


distributions = {
    "Train": train_dist,
    "Validation": validation_dist,
    "Test": test_dist
}

class_indices = sorted(list(set().union(*[d.keys() for d in distributions.values()])))
class_labels = [class_names[i] for i in class_indices]

x = np.arange(len(class_indices))  # pozicije grupa po x-osi
width = 0.25  # širina stupca

train_counts = [train_dist.get(i, 0) for i in class_indices]
val_counts = [validation_dist.get(i, 0) for i in class_indices]
test_counts = [test_dist.get(i, 0) for i in class_indices]

# Crtanje
plt.figure(figsize=(12, 6))
plt.bar(x - width, train_counts, width=width, label="Train", color='skyblue')
plt.bar(x, val_counts, width=width, label="Validation", color='orange')
plt.bar(x + width, test_counts, width=width, label="Test", color='green')

# Oznake i legenda
plt.xlabel("Klasa")
plt.ylabel("Broj primjera")
plt.title("Distribucija primjera po klasi za svaki skup")
plt.xticks(ticks=x, labels=class_labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()