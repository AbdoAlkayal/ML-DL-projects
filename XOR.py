import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def perceptron_test_multi_layer():
    def classify(x1, x2, w1, w2, bias):
        # Calculate the output of a perceptron and apply the step activation function
        x = (w1 * x1) + (w2 * x2) - bias
        return 1 if x > 0 else 0

    # Input the weights and biases for both layers
    print("Enter the weights and biases for the first perceptron:")
    try:
        w1_1 = float(input("Weight w1: "))
        w1_2 = float(input("Weight w1: "))
        bias1_1 = float(input("Bias : "))
        print('----------------------------------------------------------')
        print("Enter the weights and biases for the second perceptron:")
        w2_1 = float(input("Weight w1: "))
        w2_2 = float(input("Weight w2: "))
        bias2_1 = float(input("Bias : "))
        print('----------------------------------------------------------')
        print("Enter the weights and bias for the second layer perceptron:")
        w1_3 = float(input("Weight w1: "))
        w2_3 = float(input("Weight w2: "))
        bias2_3 = float(input("Bias : "))
        print('----------------------------------------------------------')
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Input XOR dataset points from the user
    xor_input = []
    xor_output = []

    print("Enter the XOR input points as x1,x2 (e.g., 1,0 or 0,1):")
    while True:
        point = input("Enter point (or type 'done' to finish): ")
        if point.lower() == 'done':
            break
        try:
            x1, x2 = map(int, point.split(','))
            if (x1, x2) not in xor_input:  # Ensure unique points
                xor_input.append((x1, x2))
                xor_output.append(x1 ^ x2)  # XOR operation
        except ValueError:
            print("Invalid input. Please enter in the format x1,x2.")

    # Initialize dataset for plotting and calculating outputs
    dataset = []
    for x1, x2 in xor_input:
        # First layer perceptrons
        y1 = classify(x1, x2, w1_1, w1_2, bias1_1)
        y2 = classify(x1, x2, w2_1, w2_2, bias2_1)

        # Second layer perceptron
        output = classify(y1, y2, w1_3, w2_3, bias2_3)

        dataset.append((x1, x2, y1, y2, output))

    # Plot the XOR dataset and decision boundaries
    plt.figure(figsize=(8, 6))

    # Plot the XOR points and the second layer outputs
    for x1, x2, y1, y2, output in dataset:
        color = 'blue' if output == 0 else 'red'
        marker = 'o' if output == 0 else 'x'
        plt.scatter(x1, x2, color=color, label=f'Class {int(output)}', marker=marker)

    # Create decision boundaries for both layers (Layer 1 and Layer 2)
    x_vals = np.linspace(-0.5, 1.5, 100)

    # For Layer 1 (decisions based on y1 and y2)
    # Equations for the decision boundaries of each perceptron
    y_vals_1 = (-bias1_1 - w1_1 * x_vals) / w1_2  # Boundary for the first perceptron
    y_vals_2 = (-bias2_1 - w2_1 * x_vals) / w2_2  # Boundary for the second perceptron
    plt.plot(x_vals, y_vals_1, color='green', linestyle='--', label='Layer 1: Boundary 1')
    plt.plot(x_vals, y_vals_2, color='orange', linestyle='--', label='Layer 1: Boundary 2')

    # Set labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Multi-Layer Perceptron (MLP) for XOR Classification')

    # Add grid, axes, and legend
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True)

    # Handle duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Print the table for the XOR problem with intermediate outputs using tabulate
    print("\n1ST PERCEPTRON")
    print(tabulate([(int(x1), int(x2), int(y1)) for x1, x2, y1, _, _ in dataset], headers=["X1", "X2", "Y1"],
                   tablefmt="grid"))

    print("\n2ND PERCEPTRON")
    print(tabulate([(int(x1), int(x2), int(y2)) for x1, x2, _, y2, _ in dataset], headers=["X1", "X2", "Y2"],
                   tablefmt="grid"))

    print("\nOUTPUT OF XOR")
    print(
        tabulate([(int(y1), int(y2), int(output)) for _, _, y1, y2, output in dataset], headers=["Y1", "Y2", "OUTPUT"],
                 tablefmt="grid"))

    print("\nX1  X2  Y1  Y2  Output")
    print(tabulate([(int(x1), int(x2), int(y1), int(y2), int(output)) for x1, x2, y1, y2, output in dataset],
                   headers=["X1", "X2", "Y1", "Y2", "Output"], tablefmt="grid"))

    # Show the plot
    plt.show()


# Run the multi-layer perceptron test for XOR
perceptron_test_multi_layer()
