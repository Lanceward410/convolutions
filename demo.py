import numpy as np
import matplotlib.pyplot as plt
import dill  # For saving and loading data

# Function to define x[n]
def generate_x(n_range, a):
    return a**n_range

# Function to define h[n] (rectangular pulse)
def generate_h(n_range, N):
    h = np.where((n_range >= 0) & (n_range < N), 1, 0)
    return h

# Function to compute convolution
def compute_convolution(x, h, mode='full'):
    return np.convolve(x, h, mode)

# Function to save data
def save_data(filename, data):
    with open(filename, 'wb') as file:
        dill.dump(data, file)

# Function to load data
def load_data(filename):
    with open(filename, 'rb') as file:
        return dill.load(file)

# Parameters
N = 10  # Length of rectangular window h[n]
a = 0.7  # Decay factor in x[n]
n_range = np.arange(0, 30)  # Define range for n

# Generate signals
x_n = generate_x(n_range, a)
h_n = generate_h(n_range, N)

# Compute convolution
y_n = compute_convolution(x_n, h_n, mode='full')

# Save data
save_data("conv_results.pkl", {"x_n": x_n, "h_n": h_n, "y_n": y_n})

# Define n range for y[n]
n_y = np.arange(len(y_n))

# Plot Results
plt.figure(figsize=(10, 6))

# Plot x[n]
plt.subplot(3, 1, 1)
plt.stem(n_range, x_n, linefmt='b-', markerfmt='bo', basefmt='r-', label='x[n] = a^n u[n]')
plt.title("Input Signal x[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot h[n]
plt.subplot(3, 1, 2)
plt.stem(n_range, h_n, linefmt='g-', markerfmt='go', basefmt='r-', label='h[n] = u[n] - u[n-N]')
plt.title("Impulse Response h[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot y[n]
plt.subplot(3, 1, 3)
plt.stem(n_y, y_n, linefmt='r-', markerfmt='ro', basefmt='r-', label='y[n] = x[n] * h[n]')
plt.title("Output Signal y[n] (Convolution Result)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()