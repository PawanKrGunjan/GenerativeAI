import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

fig, ax = plt.subplots()
ax.plot(np.linspace(-10, 10, 500), sigmoid(np.linspace(-10, 10, 500)), label='Sigmoid')
ax.plot(np.linspace(-10, 10, 500), sigmoid_derivative(np.linspace(-10, 10, 500)), label='Derivative of Sigmoid', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x) / f\'(x)')
plt.legend()
plt.savefig('sigmoid_plot.png')
plt.show()
