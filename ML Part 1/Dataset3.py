import numpy as np
from sklearn.linear_model import LogisticRegression

# Function to fill missing values with NaN
def replace_missing_with_nan(data):
    data[data == 1.00000000000000e+99] = np.nan
    return data

# Function to fill missing values with column means
def fill_missing_with_mean(data):
    column_means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(column_means, inds[1])
    return data

# Function to perform classification with Logistic Regression
def perform_classification_lr(train_data, train_labels, test_data):
    # Replace missing values with NaN in training data
    train_data = replace_missing_with_nan(train_data)

    # Fill missing values in training data using column means
    train_data = fill_missing_with_mean(train_data)

    # Create and fit Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)  # Adjust max_iter as needed
    lr_model.fit(train_data, train_labels)

    # Predict test labels
    test_predictions = lr_model.predict(test_data)
    return test_predictions

# Load data from text files for dataset 3
# Load data from text files for dataset 3 with comma delimiter
train_data_3 = np.loadtxt('TrainData3.txt', delimiter='\t')  
train_labels_3 = np.loadtxt('TrainLabel3.txt')  
test_data_3 = np.loadtxt('TestData3.txt', delimiter=',')  


# Perform classification for dataset 3 using Logistic Regression
test_predictions_3 = perform_classification_lr(train_data_3, train_labels_3, test_data_3)

# Write all predictions to a single file for dataset 3
with open("all_test_predictions_dataset3.txt", "w") as file:
    for prediction in test_predictions_3:
        file.write(f"{prediction}\n")
