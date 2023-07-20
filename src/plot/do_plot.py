import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('losses.csv')

sns.set_style('whitegrid')

for column in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = np.linspace(0, 1, len(df[column].dropna()))

    ax.plot(epochs, df[column].dropna(), label=f'Loss')

    ax.set_title(f'Validation Loss for {column}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')

    ax.set_xlim(0, 1)

    ax.legend()

    fig.savefig(f'losses_imgs/{column}_train_loss.png')

    plt.close(fig)
