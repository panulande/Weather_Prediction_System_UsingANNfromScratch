import numpy as np
import pandas as pd

# Define the weather dataset matrix
data = np.array([
    ["2012-01-01", 0.0, 12.8, 5.0, 4.7, "drizzle"],
    ["2012-01-02", 10.9, 10.6, 2.8, 4.5, "rain"],
    ["2012-01-03", 0.8, 11.7, 7.2, 2.3, "rain"],
    ["2012-01-04", 20.3, 12.2, 5.6, 4.7, "rain"],
    ["2012-01-05", 1.3, 8.9, 2.8, 6.1, "rain"],
    ["2012-01-06", 2.5, 4.4, 2.2, 2.2, "rain"],
    ["2012-01-07", 0.0, 7.2, 2.8, 2.3, "rain"],
    ["2012-01-08", 0.0, 10.0, 2.8, 2.0, "sun"],
    ["2012-01-09", 4.3, 9.4, 5.0, 3.4, "rain"],
    ["2012-01-10", 1.0, 6.1, 0.6, 3.4, "rain"],
    ["2012-01-11", 0.0, 6.1, -1.1, 5.1, "sun"],
    ["2012-01-12", 0.0, 6.1, -1.7, 1.9, "sun"],
    ["2012-01-13", 0.0, 5.0, -2.8, 1.3, "sun"],
    ["2012-01-14", 4.1, 4.4, 0.6, 5.3, "snow"],
    ["2012-01-15", 5.3, 1.1, -3.3, 3.2, "snow"],
    ["2012-01-16", 2.5, 1.7, -2.8, 5.0, "snow"],
    ["2012-01-17", 8.1, 3.3, 0.0, 5.6, "snow"],
    ["2012-01-18", 19.8, 0.0, -2.8, 5.0, "snow"],
    ["2012-01-19", 15.2, -1.1, -2.8, 1.6, "snow"],
    ["2012-01-20", 13.5, 7.2, -1.1, 2.3, "snow"],
    ["2012-01-21", 3.0, 8.3, 3.3, 8.2, "rain"],
    ["2012-01-22", 6.1, 6.7, 2.2, 4.8, "rain"],
    ["2012-01-23", 0.0, 8.3, 1.1, 3.6, "rain"],
    ["2012-01-24", 8.6, 10.0, 2.2, 5.1, "rain"],
    ["2012-01-25", 8.1, 8.9, 4.4, 5.4, "rain"],
    ["2012-01-26", 4.8, 8.9, 1.1, 4.8, "rain"],
    ["2012-01-27", 0.0, 6.7, -2.2, 1.4, "drizzle"],
    ["2012-01-28", 0.0, 6.7, 0.6, 2.2, "rain"],
    ["2012-01-29", 27.7, 9.4, 3.9, 4.5, "rain"]
])

# Create a DataFrame
columns = ["date", "precipitation", "temp_max", "temp_min", "wind", "weather"]
df = pd.DataFrame(data, columns=columns)

# Extract features and labels
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']].astype(float)
y = df['weather']

# Normalize input data
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# Convert labels to numerical and then one-hot encoding
class LabelEncoder:
    def fit_transform(self, y):
        classes = np.unique(y)
        self.class_mapping = {label: i for i, label in enumerate(classes)}
        self.inverse_class_mapping = {i: label for label, i in self.class_mapping.items()}
        return np.array([self.class_mapping[label] for label in y])

    def inverse_transform(self, encoded_labels):
        return np.array([self.inverse_class_mapping[int(i)] for i in encoded_labels.flatten()])



label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class OneHotEncoder:
    def fit_transform(self, y):
        num_classes = len(np.unique(y))
        result = np.zeros((len(y), num_classes))
        for i, label in enumerate(y):
            result[i, label] = 1
        return result

onehot_encoder = OneHotEncoder()
y_onehot = onehot_encoder.fit_transform(y_encoded)

# Define activation functions and loss for classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - np.tanh(x)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def softmax_prime(x):
    p = softmax(x)
    return p * (1 - p)

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

def categorical_crossentropy_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred) / len(y_true)

# Neural network classes
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.output_activation = None
        self.output_activation_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime, output_activation, output_activation_prime):
        self.loss = loss
        self.loss_prime = loss_prime
        self.output_activation = output_activation
        self.output_activation_prime = output_activation_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers[:-1]:
                output = layer.forward_propagation(output)

            # Use sigmoid activation for the output layer
            output = self.output_activation(self.layers[-1].forward_propagation(output))
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        input_size = self.layers[0].weights.shape[0]

        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                x_input = x_train[j].reshape(1, input_size)

                output = x_input
                for layer in self.layers[:-1]:
                    output = layer.forward_propagation(output)

                # Use sigmoid activation for the output layer
                output = self.output_activation(self.layers[-1].forward_propagation(output))

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                error = self.output_activation_prime(output) * error
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

# Create the neural network
num_classes = len(np.unique(y))
net = Network()
net.add(FCLayer(4, 32))  # Increased the number of neurons
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(32, num_classes))  # Increased the number of neurons
net.add(ActivationLayer(sigmoid, sigmoid_prime))
# Set the categorical crossentropy loss function
net.use(categorical_crossentropy, categorical_crossentropy_prime, sigmoid, sigmoid_prime)

# Train the neural network with more epochs
net.fit(X_normalized.values, y_onehot, epochs=500, learning_rate=0.01)

# Test the neural network
new_inputs = np.array([[13.5, 7.2, -1.1, 2.3]])
print(new_inputs.shape)
# Replace with your new inputs
new_inputs_normalized = (new_inputs - X_mean.values) / X_std.values
new_inputs_normalized = new_inputs_normalized.reshape(1, -1)
predictions = net.predict(new_inputs_normalized)
print("class_mapping:", label_encoder.class_mapping)
print("unique predicted indices:", np.unique(np.argmax(predictions[-1], axis=1)))

predicted_classes = [label_encoder.inverse_transform(i) for i in np.argmax(predictions[-1], axis=1)]
print(predicted_classes)
