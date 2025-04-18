import numpy as np
import matplotlib.pyplot as plt

def perceptron_test():

    def classify(x1, x2):
        x = (w1 * x1) + (w2 * x2) - bias
        if (x > 0):
            return 1
        else:
            return 0

    # Input the weights and bias
    print("Enter the weights and bias for the perceptron:")
    try:
        w1 = float(input("Weight w1: "))
        w2 = float(input("Weight w2: "))
        bias = float(input("Bias: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Input the dataset
    print("Enter the dataset points as x1,x2 (e.g., 1,2 or 3,4):")
    dataset = []
    while True:
        point = input("Enter point (or type 'done' to finish): ")
        if point.lower() == 'done':
            break
        try:
            x1, x2 = map(float, point.split(','))
            output = classify(x1, x2)
            dataset.append((x1, x2, output))

        except ValueError:
            print("Invalid input. Please enter in the format x1,x2.")

    dataset = np.array(dataset)

    # Plot the dataset points
    plt.figure(figsize=(8, 6))
    for x1, x2, output in dataset:
        color = 'blue' if output == 0 else 'red'
        marker = 'o' if output == 0 else 'x'
        plt.scatter(x1, x2, color=color, label=f'Class {int(output)}', marker=marker)

    # Calculate and mark the intersection points (x-axis and y-axis)
    x_intercept = bias / w1 if w1 != 0 else None  # x-intercept (-bias / w1)
    y_intercept = bias / w2 if w2 != 0 else None  # y-intercept (-bias / w2)

    # Ensure the line goes through both intercepts
    if x_intercept is not None and y_intercept is not None:
        x1, y1 = 0, x_intercept  # First point (0, 1)
        x2, y2 = y_intercept, 0  # Second point (1, 0)

        # Calculate the slope (m) of the line passing through these two points
        m = (y2 - y1) / (x2 - x1)  # Slope formula: (y2 - y1) / (x2 - x1)

        # Now that we have the slope, we use the point (0, 1) to find the intercept (c)
        c = y1 - m * x1  # y = mx + c, so c = y1 - m * x1

        # Define the decision boundary line equation: y = mx + c
        x_vals = np.linspace(min(dataset[:, 0]) - 1, max(dataset[:, 0]) + 1, 100)  # Extending range for x-axis
        y_vals = m * x_vals + c  # Equation of the line
        plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

    # Set labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Perceptron Decision Boundary with Dataset')

    # Add grid, axes, and legend
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True)

    # Handle duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Show the plot
    print("X1\tX2\tOutput")
    for x1, x2, output in dataset:
        print(f"{int(x1)}\t{int(x2)}\t{int(output)}")

    plt.show()

# Run the perceptron test
perceptron_test()
