import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.array([[1, 1], [2, -2], [-1, -1.5], [-2, -1],[-2,1],[1.5,-0.5]])  # Feature matrix
y = np.array([1, 0, 0, 0,1,1])


def perceptron_learning(X, y, max_iterations=1000, n=0.2, tolerance=1e-4):
    # Initialize weights and bias randomly
    np.random.seed(42)  # For reproducibility
    w0 = np.random.rand()
    w1 = np.random.rand()
    w2 = np.random.rand()

    errors = []
    previous_error_count = float('inf')

    for _ in range(max_iterations):
        error_count = 0
        for i in range(X.shape[0]):  # Corrected loop range
            # Calculate the perceptron output
            linear_output = w0 + w1 * X[i][0] + w2 * X[i][1]
            prediction = 1 if linear_output > 0 else 0

            # Update weights and bias if there's a misclassification
            if prediction != y[i]:
                error_count += 1
                w0 += n * (y[i] - prediction) * 1
                w1 += n * (y[i] - prediction) * X[i][0]
                w2 += n * (y[i] - prediction) * X[i][1]

        print(w0, w1, w2)

        errors.append(error_count)

        # Stop early if error count is zero (perfect classification)
        if error_count == 0:
            print(f"Stopping early after {_ + 1} iterations due to no misclassifications.")
            break

        previous_error_count = error_count

    return w0, w1, w2, errors


# Train the perceptron
w0, w1, w2, errors = perceptron_learning(X, y, max_iterations=1000, n=0.1)

# Plot decision boundary
plt.figure(figsize=(8, 6))

# Plot the data points
for i in range(X.shape[0]):
    color = 'blue' if y[i] == 0 else 'red'
    marker = 'o' if y[i] == 0 else 'x'
    plt.scatter(X[i, 0], X[i, 1], color=color, marker=marker, s=100, label=f"Class {y[i]}" if i == 0 else "")

# Calculate and plot the decision boundary
x_vals = np.linspace(-4, 4, 100)  # x-axis range for plotting the boundary
y_vals = - (w0 + w1 * x_vals) / w2  # Equation of the decision boundary: x2 = -(w0 + w1*x1) / w2

plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

# Set plot labels and title
plt.xlabel('Feature 1 (X1)')
plt.ylabel('Feature 2 (X2)')
plt.title('Perceptron Decision Boundary with Dataset')

# Add grid, axes, and legend
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True)

# Handle duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Print results
print("Final weights:", w0, w1, w2)
print("Errors at each iteration:", errors)

plt.show()
