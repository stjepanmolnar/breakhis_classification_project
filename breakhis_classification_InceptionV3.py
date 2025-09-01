import tensorflow as tf
from keras.applications import InceptionV3
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

for images, labels in dataset.take(1):
    # Get the first image and label from the batch
    image = images[0]
    label = labels[0]
    break

# plt.imshow(image.numpy().astype(int))
# plt.title(f"Label: {label.numpy()}")
# plt.axis('off')
# plt.show()


# Classification model 
continuous_training = True
if continuous_training:
    model = load_model("models/inception_fusion_model_final.keras")

     # Prvo zamrzni sve slojeve
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-17:]:
        layer.trainable = True

    for i, layer in enumerate(model.layers[-20:]):  
        print(i, layer.name, layer.trainable)
else:
    # Data augementation
    input_layer = Input(shape=(299, 299, 3))

    # Preprocessing layers
    x = layers.RandomBrightness(0.2)(input_layer)
    x = layers.RandomFlip()(x)
    x = layers.RandomRotation(0.2)(x)


    InceptionV3_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=x)

    InceptionV3_model.trainable = False
    InceptionV3_model.summary()

    # Extract intermediate layers
    layer_names = ['mixed4', 'mixed7', 'mixed10']
    intermediate_outputs = [InceptionV3_model.get_layer(name).output for name in layer_names]
    # Create a new model that outputs intermediate layers
    intermediate_model = Model(inputs=InceptionV3_model.inputs, outputs=intermediate_outputs)

    # Define the branches for each intermediate output
    branch_outputs = []
    for output in intermediate_outputs:
        x = layers.GlobalAveragePooling2D()(output)
        x = L2NormalizationLayer()(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        branch_outputs.append(x)
    # Concatenate the branch outputs for score-level fusion
    fusion = tf.keras.layers.Concatenate()(branch_outputs)

    # Final dense layers
    x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(fusion)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    final_output = layers.Dense(8, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=final_output)


stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/models/checkpoint.model.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)




model.save("models/inception_fusion_model.keras")

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Priprema svih labela u obliku indeksa klasa
all_labels = []
for _, labels in dataset_train.unbatch():
    class_index = np.argmax(labels.numpy())
    all_labels.append(class_index)

# Izračunaj težine
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(all_labels),
    y=all_labels
)

# Pretvori u rječnik
class_weights = dict(enumerate(class_weights))
print(class_weights)


model.fit(dataset_train, validation_data = dataset_validation, epochs=5, callbacks = [stopping_callback, model_checkpoint_callback], class_weight=class_weights)

model.trainable = True


model.compile(optimizer=Adam(1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


model.fit(dataset_train, validation_data = dataset_validation, epochs=5, callbacks = [stopping_callback, model_checkpoint_callback], class_weight=class_weights)
model.save("models/inception_fusion_model_final.keras")

loss, accuracy = model.evaluate(dataset_validation)
print(f"Test Accuracy: {accuracy:.4f}")

y_true = []
y_pred = []

for images, labels in dataset_validation:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))

    # Handle one-hot or sparse labels
    if labels.shape[-1] > 1:  # one-hot
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    else:  # already integer labels
        y_true.extend(labels.numpy())


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