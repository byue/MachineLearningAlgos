import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-1.0 * x))

def augment_with_intercept(dataset):
    return np.insert(dataset, 0, np.ones(shape=(1, dataset.shape[1])))

def get_initialized_weights(size):
    return np.zeros(shape=(size, 1))

def predict(dataset, weights):
    return np.rint(get_activation(dataset, weights))

def get_activation(dataset, weights):
    return sigmoid(np.dot(weights.T, dataset))

def get_cost(labels, activation):
    sample_size = labels.shape[1]
    return np.sum(labels * np.log(activation + sys.float_info.epsilon) + (1 - labels) * np.log(1 - activation + sys.float_info.epsilon)) / -sample_size

def get_weight_gradient(labels, activation, dataset):
    sample_size = labels.shape[1]
    return np.dot(dataset, (activation - labels).T) / sample_size

def train(dataset, labels, momentum=0.9, max_iterations=100000, learning_rate=0.1, epsilon=0.000001):
    print("Training with momentum={0}, max_iterations={1}, learning_rate={2}, epsilon={3}".format(momentum, max_iterations, learning_rate, epsilon))
    weights = get_initialized_weights(dataset.shape[0])
    costs = []
    prev_update_vector = None
    prev_cost = None
    for iteration in range(max_iterations):
        # forward propagation
        activation = get_activation(dataset, weights)
        cost = get_cost(labels, activation)
        costs.append(cost)

        # backward propagation
        update_vector = learning_rate * get_weight_gradient(labels, activation, dataset)

        # update weights
        if prev_update_vector is not None:
            update_vector += (momentum * prev_update_vector)
        prev_update_vector = update_vector
        weights = weights - update_vector

        # Check if we have converged
        if prev_cost is not None and abs(prev_cost - cost) < epsilon:
            print("Gradient descent has converted, exiting early after {0} iterations".format(iteration))
            break

        prev_cost = cost
    return weights, costs

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

def split_dataset(dataset, percent_train=60, shuffle=True):
    if shuffle:
        np.random.shuffle(dataset.T)

    num_samples = dataset.shape[1]
    num_train = num_samples * percent_train // 100

    train_chunk = dataset[:,:num_train]
    train_set = train_chunk[:-1]
    train_labels = train_chunk[-1][np.newaxis]

    test_chunk = dataset[:,num_train:]
    test_set = test_chunk[:-1]
    test_labels = test_chunk[-1][np.newaxis]

    return train_set, train_labels, test_set, test_labels

def get_accuracy(predictions, labels):
    num_correct = (predictions[0] == labels[0]).sum()
    accuracy = float(num_correct) / labels.shape[1]
    return accuracy

def main():
    dataset, features = preprocess_data(load_data('admissionsData.txt'))

    train_set, train_labels, test_set, test_labels = split_dataset(dataset)

    print("Number of features: {0}".format(len(features)))
    print("Train set size: {0}".format(train_set.shape[1]))
    print("Test set size: {0}".format(test_set.shape[1]))

    weights, costs = train(train_set, train_labels)

    predictions = predict(test_set, weights)

    accuracy = get_accuracy(predictions, test_labels)

    print("Accuracy: {0}".format(accuracy))

    plt.plot(costs)
    plt.show()

if __name__ == "__main__":
    main()
