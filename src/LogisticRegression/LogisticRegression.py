import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-1.0 * x))

def augment_with_intercept(dataset):
    return np.insert(dataset, 0, np.ones(shape=(1, dataset.shape[1])))

def get_initialized_weights(size):
    return np.random.uniform(low=0.0, high=1.0, size=(size, 1))

def predict(dataset, weights):
    return np.rint(get_activation(dataset, weights))

def get_activation(dataset, weights):
    return sigmoid(np.dot(weights.T, dataset))

def get_cost(labels, activation):
    sample_size = labels.shape[1]
    # we add float epsilon to handle activation being 0
    return np.sum(labels * np.log(activation + sys.float_info.epsilon) + (1 - labels) * np.log(1 - activation + sys.float_info.epsilon)) / -sample_size

def get_weight_gradient(labels, activation, dataset):
    sample_size = labels.shape[1]
    return np.dot(dataset, (activation - labels).T) / sample_size

def train(train_set, train_labels, validation_set, validation_labels, momentum=0.2, max_iterations=100000, learning_rate=0.001, epsilon=0.0000001):
    print("Training with momentum={0}, max_iterations={1}, learning_rate={2}, epsilon={3}".format(momentum, max_iterations, learning_rate, epsilon))
    weights = get_initialized_weights(train_set.shape[0])
    prev_update_vector = None
    train_costs = []
    validation_costs = []
    train_accuracies = []
    validation_accuracies = []
    prev_validation_cost = None

    for iteration in range(max_iterations):
        # Check accuracy
        validation_accuracy = get_accuracy(predict(validation_set, weights), validation_labels)
        train_accuracy = get_accuracy(predict(train_set, weights), train_labels)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        if iteration % 1000 == 0:
            print("Iteration: {0}, train Accuracy: {1}, Validation Accuracy: {2}".format(iteration, train_accuracy, validation_accuracy))

        # forward propagation
        train_activation = get_activation(train_set, weights)
        train_cost = get_cost(train_labels, train_activation)
        train_costs.append(train_cost)

        validation_cost = get_cost(validation_labels, get_activation(validation_set, weights))
        validation_costs.append(validation_cost)

        # backward propagation
        update_vector = learning_rate * get_weight_gradient(train_labels, train_activation, train_set)

        # update weights
        if prev_update_vector is not None:
            update_vector += (momentum * prev_update_vector)
        prev_update_vector = update_vector
        weights = weights - update_vector

        # Check if we have converged
        if prev_validation_cost is not None and abs(prev_validation_cost - validation_cost) < epsilon:
            print("Gradient descent has converged, exiting early after {0} iterations".format(iteration))
            break

        prev_validation_cost = validation_cost
    return weights, train_costs, train_accuracies, validation_costs, validation_accuracies

def load_data(file_path):
    return pd.read_csv(file_path, sep=',', header=0)

def normalize(feature):
    return (feature - np.mean(feature)) / np.std(feature)

def preprocess_data(df):
    # ignore serial no in first column
    df.drop(df.columns[0],axis=1,inplace=True)
    features = df.columns.values[:-1]

    # Normalize dataset
    df['GRE Score'] = normalize(df['GRE Score'])
    df['TOEFL Score'] = normalize(df['TOEFL Score'])
    df['University Rating'] = normalize(df['University Rating'])
    df['SOP'] = normalize(df['SOP'])
    df['CGPA'] = normalize(df['CGPA'])

    df_values = df.values.T
    # augment dataset with first row representing intercept with 1's
    dataset = np.insert(df_values, 0, np.ones(shape=(1, df_values.shape[1])), axis=0)
    # last row of dataset will have labels
    dataset[-1] = np.rint(dataset[-1])[np.newaxis]
    return dataset, features

def split_dataset(dataset, percent_train=60, percent_validation=20):
    num_samples = dataset.shape[1]
    num_train = num_samples * percent_train // 100
    num_validation = num_samples * percent_validation // 100

    end_of_train = num_train
    end_of_validation = num_train + num_validation

    train_chunk = dataset[:,:end_of_train]
    train_set = train_chunk[:-1]
    train_labels = train_chunk[-1][np.newaxis]
    
    validation_chunk = dataset[:,end_of_train:end_of_validation]
    validation_set = validation_chunk[:-1]
    validation_labels = validation_chunk[-1][np.newaxis]

    test_chunk = dataset[:,end_of_validation:]
    test_set = test_chunk[:-1]
    test_labels = test_chunk[-1][np.newaxis]

    return train_set, train_labels, validation_set, validation_labels, test_set, test_labels

def get_accuracy(predictions, labels):
    num_correct = (predictions[0] == labels[0]).sum()
    accuracy = float(num_correct) / labels.shape[1]
    return accuracy

def plot_costs(train_costs, train_accuracies, validation_costs, validation_accuracies):
    plt.plot(train_costs, label="Train Loss")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(validation_costs, label="Validation Loss")
    plt.plot(validation_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.show()

def main():
    dataset, features = preprocess_data(load_data('admissionsData.txt'))

    # sets are m x n matrices where m is number features + bias, n is number samples
    # first row is all 1's to accomodate biases
    train_set, train_labels, validation_set, validation_labels, test_set, test_labels = split_dataset(dataset)

    print("Number of features: {0}".format(len(features)))
    print("Train set size: {0}".format(train_set.shape[1]))
    print("Validation set size: {0}".format(validation_set.shape[1]))
    print("Test set size: {0}".format(test_set.shape[1]))

    weights, train_costs, train_accuracies, validation_costs, validation_accuracies = train(train_set, train_labels, validation_set, validation_labels)

    print("Last Train Cost: {0}".format(train_costs[-1]))
    print("Last Train Accuracy: {0}".format(train_accuracies[-1]))
    print("Last Validation Cost: {0}".format(validation_costs[-1]))
    print("Last Validation Accuracy: {0}".format(validation_accuracies[-1]))

    # Evaludate Test Set
    test_predictions = predict(test_set, weights)
    test_accuracy = get_accuracy(test_predictions, test_labels)
    print("Test Set Accuracy: {0}".format(test_accuracy))

    # Plot loss curves for train and validation set
    plot_costs(train_costs, train_accuracies, validation_costs, validation_accuracies)


if __name__ == "__main__":
    main()
