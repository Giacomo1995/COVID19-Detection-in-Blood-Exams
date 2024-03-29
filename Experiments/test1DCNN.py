# 1D-CNN

# Imports
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import seaborn as sns
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, accuracy_score
from tensorflow import keras


# Flags
USE_RAW_DATA = True
USE_TEST_SET = True
FEATURE_SELECTION = False
CUSTOM_FEATURE_SELECTION = False

COMPRESSED = True


# Read training set
print("Reading training set...")
training_set = pd.read_csv(os.path.join("Data", "training_set.csv"), compression='gzip') if COMPRESSED else pd.read_csv(os.path.join("Data", "training_set.csv"), compression=None)
print("Training set loaded correctly.")

# Read validation set
print("Reading validation set...")
validation_set = pd.read_csv(os.path.join("Data", "validation_set.csv"), compression='gzip') if COMPRESSED else pd.read_csv(os.path.join("Data", "validation_set.csv"), compression=None)
print("Validation set loaded correctly.")

# Read test set
print("Reading test set...")
test_set = pd.read_csv(os.path.join("Data", "test_set.csv"), compression='gzip') if COMPRESSED else pd.read_csv(os.path.join("Data", "test_set.csv"), compression=None)
print("Test set loaded correctly.")


print("Training set shape:", training_set.shape)
print("Validation set shape:", validation_set.shape)
if USE_TEST_SET:
    print("Test set shape:", test_set.shape)


# Conversion from pandas dataframe to numpy array
y_train = np.array(training_set["12210"]).astype(int)
del training_set["12210"]
X_train = np.array(training_set)

y_val = np.array(validation_set["12210"]).astype(int)
del validation_set["12210"]
X_val = np.array(validation_set)

if USE_TEST_SET:
    y_test = np.array(test_set["12210"]).astype(int)
    del test_set["12210"]
    X_test = np.array(test_set)


# Standardization
std_scaler = StandardScaler()
X_train_standardized = std_scaler.fit_transform(X_train)
X_val_standardized = std_scaler.transform(X_val)
if USE_TEST_SET:
    X_test_standardized = std_scaler.transform(X_test)


if FEATURE_SELECTION:
    print("Feature Selection...")

    if CUSTOM_FEATURE_SELECTION:  # Manual feature selection with 3000 features
        header = list(range(0, 12210, 4))

        if USE_RAW_DATA:
            # Raw data
            X_train = (X_train[:, header])[:, :3000]
            X_val = (X_val[:, header])[:, :3000]
            if USE_TEST_SET:
                X_test = (X_test[:, header])[:, :3000]

            print("Raw Data")
            print("X_train shape:", X_train.shape)
            print("X_val shape:", X_val.shape)
            if USE_TEST_SET:
                print("X_test shape:", X_test.shape)
            print("\n")
        else:
            # Standardized data
            X_train_standardized = (X_train_standardized[:, header])[:, :3000]
            X_val_standardized = (X_val_standardized[:, header])[:, :3000]
            if USE_TEST_SET:
                X_test_standardized = (X_test_standardized[:, header])[:, :3000]

            print("Standardized Data")
            print("X_train_standardized shape:", X_train_standardized.shape)
            print("X_val_standardized shape:", X_val_standardized.shape)
            if USE_TEST_SET:
                print("X_test_standardized shape:", X_test_standardized.shape)
    else:  # Standard Feature Selection
        if USE_RAW_DATA:
            # Raw Data
            selector = SelectKBest(mutual_info_classif, k=3000)
            X_train = selector.fit_transform(X_train, y_train)
            X_val = selector.transform(X_val)
            if USE_TEST_SET:
                X_test = selector.transform(X_test)

            print("Raw Data")
            print("X_train shape:", X_train.shape)
            print("X_val shape:", X_val.shape)
            if USE_TEST_SET:
                print("X_test shape:", X_test.shape)
            print("\n")
        else:
            # Standardized Data
            selector = SelectKBest(mutual_info_classif, k=3000)
            X_train_standardized = selector.fit_transform(X_train_standardized, y_train)
            X_val_standardized = selector.transform(X_val_standardized)
            if USE_TEST_SET:
                X_test_standardized = selector.transform(X_test_standardized)

            print("Standardized Data")
            print("X_train_standardized shape:", X_train_standardized.shape)
            print("X_val_standardized shape:", X_val_standardized.shape)
            if USE_TEST_SET:
                print("X_test_standardized shape:", X_test_standardized.shape)
            print("\n")


# Shuffle data
print("Shuffling data...")

if USE_RAW_DATA:
    # Raw data
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_val, y_val = shuffle(X_val, y_val, random_state=0)
    if USE_TEST_SET:
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
else:
    # Standardized data
    X_train, y_train = shuffle(X_train_standardized, y_train, random_state=0)
    X_val, y_val = shuffle(X_val_standardized, y_val, random_state=0)
    if USE_TEST_SET:
        X_test, y_test = shuffle(X_test_standardized, y_test, random_state=0)


try:
    X_train = X_train.reshape(X_train.shape[0], (X_train.shape[1] // 10), 10)
    X_val = X_val.reshape(X_val.shape[0], (X_val.shape[1] // 10), 10)
    X_test = X_test.reshape(X_test.shape[0], (X_test.shape[1] // 10), 10)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
except:
    sys.exit("The number of selected features must be a multiple of 10.")


leaky_relu = keras.layers.LeakyReLU()

clf = keras.models.Sequential()
clf.add(keras.layers.Conv1D(filters=32, kernel_size=7, input_shape=(1221, 10)))
clf.add(keras.layers.MaxPooling1D(7))
clf.add(keras.layers.Conv1D(filters=64, kernel_size=5))
clf.add(keras.layers.MaxPooling1D(5))
clf.add(keras.layers.Conv1D(filters=128, kernel_size=3))
clf.add(keras.layers.MaxPooling1D(3))
clf.add(keras.layers.Flatten())
clf.add(keras.layers.Dense(1000, activation='relu'))
clf.add(keras.layers.Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, name='Adam')
clf.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
clf.summary()

num_epochs = 1000
history = clf.fit(X_train, y_train, epochs=num_epochs, batch_size=128, validation_data=(X_val, y_val))


try:
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
except:
    pass


# Results
print("Training set outcome")
train_predictions = np.round(clf.predict(X_train))
print(classification_report(y_train, train_predictions))
print("Confusion matrix\n", confusion_matrix(y_train, train_predictions))
print("\n")

print("Validation set outcome")
val_predictions = np.round(clf.predict(X_val))
print(classification_report(y_val, val_predictions))
print("Confusion matrix\n", confusion_matrix(y_val, val_predictions))
print("\n")

if USE_TEST_SET:
    print("Test set outcome")
    test_predictions = np.round(clf.predict(X_test))
    print(classification_report(y_test, test_predictions))
    print("Confusion matrix\n", confusion_matrix(y_test, test_predictions))
