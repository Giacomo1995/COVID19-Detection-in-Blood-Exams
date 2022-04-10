# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm, neighbors, naive_bayes, neural_network, ensemble
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from joblib import dump
from tensorflow import keras


CUSTOM = True  # Whether to select a custom model or not


# Reading training set
print("Reading training set...")
training_set = pd.read_csv(os.path.join('Data', 'training_set.csv'), compression=None)
print("Training set loaded correctly.")

# Reading validation set
print("Reading validation set...")
validation_set = pd.read_csv(os.path.join('Data', 'validation_set.csv'), compression=None)
print("Validation set loaded correctly.")

# Training set conversion
y_train = np.array(training_set['12210']).astype(int)
del training_set['12210']
X_train = np.array(training_set)

# Validation set conversion
y_val = np.array(validation_set['12210']).astype(int)
del validation_set['12210']
X_val = np.array(validation_set)

# Standardization
print("Standardization...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Feature Selection...")
#selector = SelectKBest(f_classif, k=3000)
#selector = SelectKBest(chi2, k=3000)
selector = SelectKBest(mutual_info_classif, k=3000)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)

# print("Manual Feature Selection...")
# header = list(range(0, 12210, 4))
# X_train = (X_train[:, header])[:, :3000]
# X_val = (X_val[:, header])[:, :3000]

# Save selector
print("Saving selector...")
dump(selector, open('selector.pkl', 'wb'))
print("Selector saved correctly.")
#load(open('selector.pkl', 'rb'))

# Shuffle data
print("Shuffling data...")
X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_val, y_val = shuffle(X_val, y_val, random_state=0)


if CUSTOM:
    leaky_relu = keras.layers.LeakyReLU()
    clf = keras.models.Sequential()
    clf.add(keras.layers.Dropout(rate=0.1, seed=0, input_shape=(X_train.shape[1],)))
    clf.add(keras.layers.Dense(300, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dropout(rate=0.1, seed=0))
    clf.add(keras.layers.Dense(200, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dropout(rate=0.1, seed=0))
    clf.add(keras.layers.Dense(100, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dropout(rate=0.1, seed=0))
    clf.add(keras.layers.Dense(75, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dropout(rate=0.1, seed=0))
    clf.add(keras.layers.Dense(50, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dropout(rate=0.1, seed=0))
    clf.add(keras.layers.Dense(25, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dropout(rate=0.1, seed=0))
    clf.add(keras.layers.Dense(5, activation=leaky_relu, kernel_initializer=keras.initializers.he_uniform(seed=0)))
    clf.add(keras.layers.Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, name='Adam')
    clf.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    clf.summary()

    print("Training...")
    num_epochs = 1000
    history = clf.fit(X_train, y_train, epochs=num_epochs, batch_size=128, validation_data=(X_val, y_val))

    print("Saving the model...")
    clf.save(os.path.join('Models', 'custom'))
    print("Model saved correctly.")

    # Validate the model
    print("Validating the model...")
    results = clf.evaluate(X_val, y_val)
    print("Validation loss, validation acc:", results)

    y_pred = np.round(clf.predict(X_val))
    print(classification_report(y_val, y_pred))

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
else:
    print("Training...")
    #clf = ensemble.RandomForestClassifier(n_estimators=1000)
    clf = svm.SVC(kernel='rbf', gamma='scale', C=10)
    clf.fit(X_train, y_train)

    print("Computing accuracy on training set...")
    train_predictions = clf.predict(X_train)
    print("Training set accuracy: ", accuracy_score(y_train, train_predictions))

    print("Computing accuracy on validation set...")
    val_predictions = clf.predict(X_val)
    print("Validation set accuracy: ", accuracy_score(y_val, val_predictions))
