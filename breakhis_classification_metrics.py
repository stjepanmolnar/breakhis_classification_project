import tensorflow as tf
import keras
from keras.applications import EfficientNetB3
from keras import layers, models, Input, Model
from keras.layers import Average
from keras.optimizers import Adam
from keras.saving import load_model, register_keras_serializable
import matplotlib.pyplot as plt
import collections
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
keras.config.enable_unsafe_deserialization()

#Custom layer
@register_keras_serializable(package="Custom", name="L2NormalizationLayer")
class L2NormalizationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


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
    image_size=(299, 299),
    batch_size=32,
    label_mode="categorical"
)

dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(299, 299),
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

model_EfficientNetB3 = load_model("models/efficientnet_fusion_model_final.keras", custom_objects={"L2NormalizationLayer": L2NormalizationLayer})

model_InceptionV3 = load_model("models/inception_fusion_model_final.keras", custom_objects={"L2NormalizationLayer": L2NormalizationLayer})


models = [model_InceptionV3]

for model in models:
    loss, accuracy = model.evaluate(dataset_validation)
    print(f"Test Accuracy: {accuracy:.4f}")

    y_true = []
    y_pred = []
    images_all = []

    for images, labels in dataset_validation:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))

        # Handle one-hot or sparse labels
        if labels.shape[-1] > 1:  # one-hot
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        else:  # already integer labels
            y_true.extend(labels.numpy())

        images_all.extend(images.numpy())


    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    images_all = np.array(images_all)
    
    
    wrong_1_as_5 = np.where((y_true == 1) & (y_pred == 5))[0]
    wrong_5_as_1 = np.where((y_true == 5) & (y_pred == 1))[0]


    # Prikaz prvih 4 pogrešno klasificirane slike
    f, axarr = plt.subplots(2,2, figsize=(8,8))

    if len(wrong_1_as_5) > 0:
        axarr[0,0].imshow(images_all[wrong_1_as_5[0]].astype("uint8"))
        axarr[0,0].set_title("True 1 -> Pred 5")
        axarr[0,0].axis("off")
        
    if len(wrong_5_as_1) > 0:
        axarr[0,1].imshow(images_all[wrong_5_as_1[0]].astype("uint8"))
        axarr[0,1].set_title("True 5 -> Pred 1")
        axarr[0,1].axis("off")
        
    # Ako želiš još dvije za primjer, možeš duplicirati ili uzeti druge indekse
    if len(wrong_1_as_5) > 1:
        axarr[1,0].imshow(images_all[wrong_1_as_5[1]].astype("uint8"))
        axarr[1,0].set_title("True 1 -> Pred 5")
        axarr[1,0].axis("off")
        
    if len(wrong_5_as_1) > 1:
        axarr[1,1].imshow(images_all[wrong_5_as_1[1]].astype("uint8"))
        axarr[1,1].set_title("True 5 -> Pred 1")
        axarr[1,1].axis("off")

    plt.tight_layout()
    plt.show()

    y_true = []
    y_pred = []
    for images, labels in dataset_test:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))

        # Handle one-hot or sparse labels
        if labels.shape[-1] > 1:  # one-hot
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        else:  # already integer labels
            y_true.extend(labels.numpy())


    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on test')
    plt.show()