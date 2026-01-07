import numpy as np
import matplotlib.pyplot as plt

# 1. Define the range for x-values
# From -2π to 2π, with a sufficient number of points for a smooth curve
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)

# 2. Calculate the y-values for each wave
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)

# 3. Handle asymptotes for the tangent function
# Replace very large/small tangent values with NaN to prevent drawing misleading vertical lines
# We'll set a threshold, e.g., anything beyond -10 and 10
y_tan[y_tan > 10] = np.nan
y_tan[y_tan < -10] = np.nan

# 4. Create the plot
plt.figure(figsize=(12, 6)) # Set the figure size for better readability

# Plot Sine wave
plt.plot(x, y_sin, color='blue', linestyle='-', label='Sine (sin(x))')

# Plot Cosine wave
plt.plot(x, y_cos, color='red', linestyle='--', label='Cosine (cos(x))')

# Plot Tangent wave
plt.plot(x, y_tan, color='green', linestyle=':', label='Tangent (tan(x))')

# 5. Add titles, labels, legend, and grid
plt.title('Sine, Cosine, and Tangent Waves from -2π to 2π')
plt.xlabel('X-axis (radians)')
plt.ylabel('Y-axis')
plt.legend() # Display the legend to identify each wave
plt.grid(True, linestyle='--', alpha=0.7) # Add a grid for easier reading

# 6. Customize x-axis ticks to show multiples of pi
x_ticks = [-2 * np.pi, -3 * np.pi / 2, -np.pi, -np.pi / 2, 0,
           np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
x_labels = ['-2π', '-3π/2', '-π', '-π/2', '0',
            'π/2', 'π', '3π/2', '2π']
plt.xticks(x_ticks, x_labels)

# 7. Set y-axis limits to better visualize the waves (especially after handling tangent)
plt.ylim(-2.5, 2.5) # Adjust as needed to focus on the main parts of the waves

# 8. Adjust layout to prevent labels from overlapping
plt.tight_layout()

# 9. Save the plot as a PNG file
plt.savefig('sine_wave.png')

# 10. Display the plot
plt.show()

print("Plot saved as sine_wave.png")