import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -2π to 2π with 200 points for smooth curve
x = np.linspace(-2 * np.pi, 2 * np.pi, 200)

# Compute sine values
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='blue', linewidth=2, label='sin(x)')

# Customize the plot
plt.title('Sine Wave from -2π to 2π')
plt.xlabel('x (radians)')
plt.ylabel('sin(x)')
plt.grid(True, which='both')
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.legend()
plt.xlim(-2 * np.pi, 2 * np.pi)
plt.ylim(-1.2, 1.2)

# Save the plot as PNG
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Display the plot (optional, comment out if not needed)
plt.show()