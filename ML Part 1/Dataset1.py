import numpy as np
from sklearn import svm

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

# Function to perform classification with SVM
def perform_classification(train_data, train_labels, test_data):
    # Replacing missing values with NaN
    train_data = replace_missing_with_nan(train_data)
    test_data = replace_missing_with_nan(test_data)

    # Filling missing values in test data using column means from training data
    train_data = fill_missing_with_mean(train_data)
    test_data = fill_missing_with_mean(test_data)

    # Creating and fitting SVM model
    svm_model = svm.SVC()
    svm_model.fit(train_data, train_labels)

    # Predicting test labels
    test_predictions = svm_model.predict(test_data)
    return test_predictions

train_data_1 = np.loadtxt('TrainData1.txt') 
train_labels_1 = np.loadtxt('TrainLabel1.txt') 
test_data_1 = np.loadtxt('TestData1.txt')  

test_predictions_1 = perform_classification(train_data_1, train_labels_1, test_data_1)

with open("BrienKapriClassification1_pred.txt", "w") as file:
    for prediction in test_predictions_1:
        file.write(f"{prediction}\n")


