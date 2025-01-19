import matplotlib.pyplot as plt
import numpy as np

def say_hello():
    return "Hello, big guy"

def plot_complex_graphic():
    # Create some data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot sine wave
    plt.plot(x, y1, label='Sine Wave', color='blue', linestyle='-', marker='o')

    # Plot cosine wave
    plt.plot(x, y2, label='Cosine Wave', color='red', linestyle='--', marker='x')

    # Add title and labels
    plt.title("Sine and Cosine Waves")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Add legend
    plt.legend()

    # Save the plot as a JPG file
    plt.savefig("complex_plot.jpg")

    # Show the plot
    plt.show()

# Simple test
if __name__ == "__main__":
    assert say_hello() == "Hello, big guy"
    print("Test passed!")
    plot_complex_graphic()