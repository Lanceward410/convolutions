# Author: Lance Ward
# This script performs the convolution of x[n]*h[n] = y[n], plotting all three in discrete time
# with values x[n]=(a^n)*u[n],  h[n]=u[n]-u[n-N],  y[n]=numpy.convolve(x[n], h[n])

# It presents you with an animation of all three plots with values of a between 0.000 and 1.000

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function x[n]
def generate_x(a):
    full_n_range = np.arange(0, 1000)  # Simulating an "infinite" signal for input x[n]
    return a**full_n_range

# Function h[n]
def generate_h(n_range, N):
    h = np.where((n_range >= 0) & (n_range < N), 1, 0)
    return h

# Tunable Parameters
N = 30  # Fixed N value
colors = ["b", "r", "m"] # Color of plots x[n], h[n], and y[n]
num_total_frames = 180  # Total number of Frames, affects speed and smoothness
animation_fps = 100 # Framerate of saved file = fps
display_plot = True # Display plot at runtime? (True/False)
save_gif = False # Save plot as .gif at runtime? (True/False)
save_file_name = "convolution_N_is_30.gif" # Name of .gif that will be generated + saved in parent directory

# Generate logarithmically spaced a_values
exp_factor = 1.2  # Exponential scaling factor of increasing 'a' values, Default = 1.2
raw_indices = np.linspace(0, (1+(1000/num_total_frames)), num_total_frames)
a_values = 1 - np.exp(-exp_factor * raw_indices)
n_range = np.arange(0, N*3) # Define the DISPLAYED range of all plots (NOT the real signal range)

# Initialize figure and subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.5) # Padding

# Titles for each subplot (now right-aligned)
titles = [
    "Input Signal x[n] = (a^n) . u[n]",
    "Impulse Response h[n] = u[n] - u[n-N]",
    "Output y[n] = x[n] * h[n] (Convolution)"
]

# Initialize stem plots without data
stems = []
for ax, title, color in zip(axes, titles, colors):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10, loc='right')
    ax.set_xlabel("n", fontweight='bold')
    ax.set_ylabel("Amplitude", fontweight='bold')
    ax.grid(True)
    ax.axhline(0, color='k', linestyle='-', linewidth=1)  
    stem = ax.stem([0], [0], linefmt=color, markerfmt=color + 'o', basefmt=color)  
    stems.append(stem)

# Add a Legend to display 'a' and 'N' values
key_text = fig.text(
    0.05, 0.92,  # Adjust position of Legend here
    f"a = {a_values[0]:.3f}\nN = {N}",  # This determines precision of 'a' value
    fontsize=14,
    fontweight='bold',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.2)
)

# Animation update function
def update(frame):
    a = a_values[frame]  # Cycles through values of 'a' from 0.000 to 1.000
    x_full = generate_x(a)  # Generate full x[n] signal with 'n' values 0 to 1000
    x_n = x_full[:(N*3)]  # x_n is plotted, but x_full is used for convolution
    h_n = generate_h(np.arange(0, len(x_full)), N) # h_n must exist for range of x_n to convolute
    y_n = np.convolve(x_full, h_n, mode='full')  # Compute convolution from n=0 to n=1000
    n_y = np.arange(len(y_n))  # Full length of convolution

    # Update each subplot
    data = [(n_range, x_n), (n_range, h_n[:(N*3)]), (n_y[:(N*3)], y_n[:(N*3)])]  # Only display y[n] for N*3 values of 'n'
    for ax, stem, (n_vals, y_vals), title, color in zip(axes, stems, data, titles, colors):
        ax.clear()
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10, loc='right')
        ax.set_xlabel("n", fontweight='bold')
        ax.set_ylabel("Amplitude", fontweight='bold')
        ax.grid(True)
        ax.axhline(0, color='k', linestyle='-', linewidth=1)  
        ax.stem(n_vals, y_vals, linefmt=color, markerfmt=color + 'o', basefmt=color)  

    # Update the legend in the top-left
    key_text.set_text(f"a = {a:.3f}\nN = {N}")
    # Update formula in Title of x[n] plot, for increasing values of 'a'
    axes[0].set_title(f"Input Signal x[n] = {a:.3f}^n u[n]", fontsize=14, fontweight='bold', pad=10, loc='right')

# Create animation
if save_gif or display_plot:
    ani = animation.FuncAnimation(fig, update, frames=len(a_values), interval=round(1000/animation_fps), repeat=True)

# Save animation as .gif
if save_gif:
    print("Please wait, generating .gif ....")
    ani.save(save_file_name, writer="pillow", fps=animation_fps)
    print(".gif Saved to parent directory!")

# Displays MatPlotLib animation
if display_plot:
    plt.show()

if not save_gif and not display_plot:
    print("\nNo actions specified,")
    print("Please set a value of True for either display_plot or save_gif.\n")