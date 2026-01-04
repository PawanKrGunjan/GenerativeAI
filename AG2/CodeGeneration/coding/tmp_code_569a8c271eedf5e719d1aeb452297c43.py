import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -2π to 2π
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Calculate sine, cosine, and tangent waves
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_sin, color='blue', label='sin(x)', linewidth=2)
plt.plot(x, y_cos, color='red', label='cos(x)', linewidth=2)
plt.plot(x, y_tan, color='green', label='tan(x)', linewidth=2)

# Customize the plot
plt.xlabel('x (radians)')
plt.ylabel('Amplitude')
plt.title('Sine, Cosine, and Tangent Waves (-2π to 2π)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-3, 3)  # Limit y-axis to better view the main waves (tan has asymptotes)

# Save the plot
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Display the plot (optional)
plt.tight_layout()
plt.show()