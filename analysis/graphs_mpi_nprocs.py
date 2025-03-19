import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("medie_mpi.csv")
algorithms = df['algo'].unique()

# Color palette
colors = ['#8bd0c2', '#3eb59d', '#213c4c',
          '#00202e', '#003f5c', '#8a508f',
          '#bc5090', '#ff8531', '#ffa600']

# Graph for each algo
for algo in algorithms:
    algo_df = df[df['algo'] == algo]

    width_px = 950
    dpi = 150
    width_inch = width_px / dpi

    plt.figure(figsize=(width_inch, 600/dpi))
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    thread_values = sorted(algo_df['procs'].unique())
    log2_thread_values = np.log2(thread_values)

    plt.xticks(log2_thread_values, labels=thread_values, fontsize=12)

    for i, matrice in enumerate(algo_df['matrix'].unique()):
        matrice_df = algo_df[algo_df['matrix'] == matrice]


        log2_matrix_size = int(np.log2(matrice))  # Get log2 of the matrix size

        plt.plot(np.log2(matrice_df['procs']), matrice_df['time'],
                 label=log2_matrix_size,
                 color=colors[i % len(colors)])


    plt.xlabel('mpi processes', fontsize=14)
    plt.ylabel('wall-clock time (seconds)', fontsize=14)
    plt.legend(title='matrix 2-pow size', fontsize=10, ncol=2)


    plt.savefig(f'{algo}_graph.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.close()

print("Graphs created")
