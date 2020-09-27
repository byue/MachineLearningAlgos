import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def augment_with_intercept(dataset):
    return np.insert(dataset, 0, np.ones(shape=(1, dataset.shape[1])))

def get_initialzied_weights():
    return np.zeros(shape=(x.shape[0], 1))

def predict(dataset, weights):
    return np.rint(np.dot(dataset, weights))

def train(dataset, labels, iterations=100, learning_rate=0.5):
    num_samples = labels.shape[1]
    weights = get_initialzied_weights()
    for i in range(iterations):
        # forward propagation
        activation = sigmoid(np.dot(weights.T, dataset))
        cost = np.sum(labels * np.log(activation) + (1 - labels) * np.log(1 - activation)) / -num_samples
        # backward propagation
        weight_gradient = np.dot(dataset, (activation - labels).T) / num_samples
        weights = weights - learning_rate * weight_gradient
        print("Iteration: {0}, Cost: {1}\n".format(i, cost))
    return weights

def load_dataset():
    raise NotImplementedError

def preprocess_data(dataset):
    # augment dataset with row representing intercept with 1's
    return np.insert(dataset, 0, np.ones(shape=(1, dataset.shape[1])))

def main():
    raw_data, labels = load_dataset()
    dataset = preprocess_data(raw_data)
    weights = train(dataset, labels)
    predictions = predict(dataset, weights)
    print("Predictions: {0}".format(predictions))

if __name__ == "__main__":
    main()
