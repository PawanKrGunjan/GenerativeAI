import numpy as np
import matplotlib.pyplot as plt

# 1. Define the range for x-values
# We'll generate 500 points between -2π and 2π for a smooth curve.
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)

# 2. Calculate the y-values for each trigonometric function
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)

# 3. Create the plot
plt.figure(figsize=(12, 6)) # Set the figure size for better readability

# Plot Sine wave
plt.plot(x, y_sin, label='Sine (sin(x))', color='blue', linestyle='-')

# Plot Cosine wave
plt.plot(x, y_cos, label='Cosine (cos(x))', color='red', linestyle='--')

# Plot Tangent wave
# The tangent function has asymptotes. To prevent Matplotlib from drawing
# vertical lines connecting these discontinuities, we can filter out
# points where the tangent value is extremely large or small.
# A common approach is to set a y-limit for the plot.
plt.plot(x, y_tan, label='Tangent (tan(x))', color='green', linestyle=':')

# 4. Customize the plot for better clarity

# Title and labels
plt.title('Trigonometric Waves: Sine, Cosine, and Tangent', fontsize=16)
plt.xlabel('X-axis (radians)', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)

# Add a legend to identify each wave
plt.legend(loc='upper right', fontsize=10)

# Add a grid for easier reading of values
plt.grid(True, linestyle='--', alpha=0.7)

# Set x-axis ticks to be in terms of pi
tick_positions = [-2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
tick_labels = ['-2π', '-3π/2', '-π', '-π/2', '0', 'π/2', 'π', '3π/2', '2π']
plt.xticks(tick_positions, tick_labels, fontsize=10)

# Set y-axis limits to handle the asymptotes of the tangent function
# This prevents the plot from being stretched vertically by extreme tan values.
plt.ylim(-2.5, 2.5) # A reasonable range to show the main features of all three waves

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# 5. Save the plot
plt.savefig('sine_wave.png', dpi=300) # dpi=300 for higher resolution

# 6. Display the plot (optional, but good for immediate viewing)
plt.show()

print("Plot saved as sine_wave.png")