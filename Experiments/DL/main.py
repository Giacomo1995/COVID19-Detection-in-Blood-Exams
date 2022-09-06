# Imports
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from classifier import *
from utils import *


STRATEGY = 5  # Select the network architecture to adopt for the training process
MODEL_NAME = "model"  # Name of the model


def main():
    # Load training set and validation set
    print("Reading training set and validation set...")

    # Load the appropriate dataset based on STRATEGY value
    if STRATEGY == 1 or STRATEGY == 2 or STRATEGY == 5:
        # Read data
        training_set = pd.read_csv(os.path.join('Data', 'training_set.csv'), compression=None)
        validation_set = pd.read_csv(os.path.join('Data', 'validation_set.csv'), compression=None)
    elif STRATEGY == 3 or STRATEGY == 4:
        # Read data
        training_set = pd.read_csv(os.path.join('Data', 'preprocessed_training_set.csv'), compression=None)
        validation_set = pd.read_csv(os.path.join('Data', 'preprocessed_validation_set.csv'), compression=None)

    print("Training set and validation set loaded correctly.")

    # y_train
    y_train = np.array(training_set['12210']).astype(int)

    # y_val
    y_val = np.array(validation_set['12210']).astype(int)

    # X_train
    del training_set['12210']
    X_train = np.array(training_set)

    # X_val
    del validation_set['12210']
    X_val = np.array(validation_set)

    # Standardization step (zero mean and unit standard deviation)
    if STRATEGY == 1 or STRATEGY == 2 or STRATEGY == 5:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

    # Shuffle data
    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    X_val, y_val = shuffle(X_val, y_val, random_state=1)

    # Get the selected model based on STRATEGY
    clf = Classifier(STRATEGY)
    model = clf.model

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, name="Adam")
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    num_epochs = 1000
    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=128, validation_data=(X_val, y_val))

    if MODEL_NAME != "":
        model.save(os.path.join('Models', MODEL_NAME))  # Save the model as MODEL_NAME

    # Validate the model
    print("Validating the model...")
    results = model.evaluate(X_val, y_val, batch_size=128)
    print("validation loss, validation acc:", results)

    # Plot loss and accuracy
    epochs = range(num_epochs)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Loss plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(range(num_epochs), acc, 'b', label='Training accuracy')
    plt.plot(range(num_epochs), val_acc, 'r', label='Validation accuracy')
    plt.title('Accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
