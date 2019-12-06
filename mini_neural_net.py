import numpy as np 

def signoid(x):
    return 1 / (1+np.exp(-x))

def signoid_derivative(x):
    return x * (1-x)

training_input = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_output = np.array([[0,1,1,0]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3,1))-1

print("Random strating synaptic weights")
print(synaptic_weights)

for i in range(1000):

    input_layer = training_input
    outputs = signoid(np.dot(input_layer, synaptic_weights))

    error = training_output - outputs

    adjustments = error * signoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

    print('Synaptic weights after training')
    print(synaptic_weights)

    print('Outputs after training: ')
    print(outputs)
    print(i)
