import matplotlib.pyplot as plt
import numpy as np

# Define the data for the chart as provided in the image
quarters = ['512', '1024', '2048', '4096']
sequential_strassen_time = np.array([0.411067, 2.89363, 20.4771, 142.139])
openmp_time = np.array([0.030936, 0.17525, 1.1065, 7.3549])
mpi_time = np.array([0.0076888, 0.043156, 0.27989, 1.7324])
cuda_time = np.array( [0.264262, 0.297811, 0.6145, 2.04965])
hybrid_time = np.array([0.064477, 0.21221, 1.1824, 7.6606])

# The x position of bars
bar_width = 0.15
x = np.arange(len(quarters))

# Plotting the bar chart
fig, ax = plt.subplots()
ax.bar(x - bar_width*2, sequential_strassen_time/sequential_strassen_time, width=bar_width, label='Sequential Strassen')
ax.bar(x - bar_width, sequential_strassen_time/openmp_time, width=bar_width, label='OpenMP Strassen')
ax.bar(x, sequential_strassen_time/mpi_time, width=bar_width, label='MPI Strassen')
ax.bar(x + bar_width, sequential_strassen_time/cuda_time, width=bar_width, label='CUDA strassen')
ax.bar(x + bar_width*2, sequential_strassen_time/hybrid_time, width=bar_width, label='Hybrid (OpenMP + MPI) Strassen')

# Adding labels and title
ax.set_xlabel('Size of input matrix')
ax.set_ylabel('Speedup')
ax.set_title('Speedup of all algorithms for different matriz size')
ax.set_xticks(x + bar_width/2)
ax.set_xticklabels(quarters)
ax.legend()

# Display the plot'
plt.savefig("result.png")
