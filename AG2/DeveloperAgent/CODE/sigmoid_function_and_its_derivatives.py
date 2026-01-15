import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-10, 10, 400)
x_deriv = np. linspace(-10, 10, 399)
plt.plot(x, sigmoid(x), label='sigmoid function')
plt.plot(x_deriv, derive_sigmoid(x_deriv), label='derived sigmoid function')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sigmoid Function and its Derivative')
plt.legend()
plt.savefig('sigmoid_function_derivatived_graph.png', dpi=300)
plt.show()