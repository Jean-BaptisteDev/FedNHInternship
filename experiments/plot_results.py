import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results(result_dir):
    all_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(f) for f in all_files]
    return pd.concat(df_list, ignore_index=True)

def plot_results(df, metric, title, output_file):
    df_grouped = df.groupby(['round', 'strategy'])[metric].mean().unstack()
    df_grouped.plot()
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(metric)
    plt.legend(title='Strategy')
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    result_dir = './results'
    df = load_results(result_dir)
    plot_results(df, 'accuracy', 'Accuracy vs. Rounds', 'accuracy_vs_rounds.png')
