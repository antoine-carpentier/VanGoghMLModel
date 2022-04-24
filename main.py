import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import seaborn as sns


def tf_model():
    # first let's process our data into usable datasets
    # our data folder is structure the following way:
    # VanGoghML
    #   Van Gogh
    #       img1.png
    #       img2.png
    #       ...
    #    Not Van Gogh
    #       img1.png
    #       img2.png
    #       ...

    # We can hence use the "image_dataset_from_directory" function from keras to quickly turn our folders into datasets

    batch_size = 32
    img_height = 180
    img_width = 180

    art_directory = 'C:/Users/elver/Documents/Art/VanGoghML'

    train_ds = tf.keras.utils.image_dataset_from_directory(
        art_directory,
        validation_split=0.3,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        art_directory,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names  # classes are "Van Gogh" and "Not Van Gogh"
    num_classes = len(class_names)  # = 2

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # RGB values are between 0 and 255.
    # We want our features to be between 0 and 1
    normalization_layer = layers.Rescaling(1. / 255)

    # our datasets are relatively small (681 images total)
    # we can increase the diversity of our dataset by augmenting it
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model_count = 20
    model_list = [0] * model_count
    history = [0] * model_count

    for i in range(model_count):
        # we can now put together the different layers of our NN
        model_list[i] = Sequential([
            data_augmentation,
            normalization_layer,

            layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, kernel_size = 5, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),

            layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, kernel_size = 5, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),

            layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),  # Flatten
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)  # Output layer
        ])

        model_list[i].compile(optimizer='adam',
                              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # true first model
                              metrics=['accuracy'])

    # when loss stops improving, stop the model
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    epochs = 100

    for i in range(len(model_list)):
        history[i] = model_list[i].fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[callback],
            verbose=0
        )

        epochs_count = len(history[i].history['loss'])

        acc = max(history[i].history['accuracy'])
        val_acc = max(history[i].history['val_accuracy'])

        print(f'CNN {i + 1}: epochs count = {epochs_count}, train accuracy = {acc}, validation accuracy = {val_acc} ')

    # aggregate the results
    y_pred = model_list[0].predict(val_ds)

    predictions = np.zeros((len(y_pred), 2))
    for i in range(model_count):
        predictions = predictions + model_list[i].predict(val_ds)

    predictions = np.argmax(predictions, axis=1)
    val_labels = np.concatenate([y for x, y in val_ds], axis=0)

    # plot the confusion matrix and display various metrics about the CNN
    confusion_mtx = tf.math.confusion_matrix(val_labels, predictions)
    true_neg = confusion_mtx[0, 0]
    false_pos = confusion_mtx[0, 1]
    false_neg = confusion_mtx[1, 0]
    true_pos = confusion_mtx[1, 1]

    accuracy = (true_pos + true_neg) / (tf.reduce_sum(confusion_mtx).numpy())
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    f_one = 2 * precision * recall / (precision + recall)

    plt.figure(constrained_layout=False, tight_layout=True)
    sns.heatmap(confusion_mtx,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    print(f'Accuracy = {accuracy:.2%}\r\nPrecision = {precision:.2%}\r\nRecall = {recall:.2%}\r\nF1 = {f_one:.2%}')


if __name__ == '__main__':
    tf_model()
