import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

# Function to perform classification with Random Forest
def perform_classification_rf(train_data, train_labels, test_data):
    # Replacing missing values with NaN in training data
    train_data = replace_missing_with_nan(train_data)

    # Filling missing values in training data using column means
    train_data = fill_missing_with_mean(train_data)

    # Creating and fit Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(train_data, train_labels)

    # Predict test labels
    test_predictions = rf_model.predict(test_data)
    return test_predictions


train_data_3 = np.loadtxt('TrainData3.txt', delimiter='\t')  
train_labels_3 = np.loadtxt('TrainLabel3.txt')  
test_data_3 = np.loadtxt('TestData3.txt', delimiter=',')  



test_predictions_3 = perform_classification_rf(train_data_3, train_labels_3, test_data_3)


with open("BrienKapriClassification3.txt", "w") as file:
    for prediction in test_predictions_3:
        file.write(f"{prediction}\n")
