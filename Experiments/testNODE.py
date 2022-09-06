# NODE

# Imports
import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from pytorch_tabular import TabularModel
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


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


training_set = pd.DataFrame(X_train)
training_set['12210'] = y_train

validation_set = pd.DataFrame(X_val)
validation_set['12210'] = y_val

if USE_TEST_SET:
    test_set = pd.DataFrame(X_test)
    test_set['12210'] = y_test


# Configuration
data_config = DataConfig(
    target=['12210'],
    continuous_cols=list(training_set.columns[:-1]),
    categorical_cols=['12210']
)

trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=128,
    max_epochs=100
)

optimizer_config = OptimizerConfig()

model_config = NodeConfig(task="classification")

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

print("Training...")
tabular_model.fit(train=training_set, validation=validation_set)

result = tabular_model.evaluate(test_set)
print(result)


# Results
print("Training set outcome")
train_predictions = np.array((tabular_model.predict(training_set)).prediction)
print(classification_report(y_train, train_predictions))
print("Confusion matrix\n", confusion_matrix(y_train, train_predictions))
print("\n")

print("Validation set outcome")
val_predictions = np.array((tabular_model.predict(validation_set)).prediction)
print(classification_report(y_val, val_predictions))
print("Confusion matrix\n", confusion_matrix(y_val, val_predictions))
print("\n")

print("Test set outcome")
test_predictions = np.array((tabular_model.predict(test_set)).prediction)
print(classification_report(y_test, test_predictions))
print("Confusion matrix\n", confusion_matrix(y_test, test_predictions))
