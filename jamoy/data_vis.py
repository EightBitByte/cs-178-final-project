# data_vis.py
# 
# Used to import and visualize the dataset.
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import colorama

def plot_box_plot(train_dataset, filename='box_plot.png'):
    """Plots a boxplot of the length of the texts against its assigned class."""
    texts = [example['text'] for example in train_dataset]
    labels = [example['label'] for example in train_dataset]

    string_lengths = [len(text) for text in texts]

    lengths_by_class = {0: [], 1: []}
    unique_labels = [0, 1]
    display_labels = ['Negative', 'Positive']

    for i, label in enumerate(labels):
        lengths_by_class[label].append(string_lengths[i])
    
    plot_data = [lengths_by_class[label] for label in unique_labels]

    plt.boxplot(plot_data, tick_labels=display_labels, patch_artist=True)

    plt.title('Distribution of Review Lengths by Class (Box Plot)')
    plt.xlabel('Class Label')
    plt.ylabel('String Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'./plots/{filename}')


def plot_histogram(train_dataset, filename='histogram.png'):
    """Plots a histogram of the length of the texts against its assigned class."""
    texts = [example['text'] for example in train_dataset]
    labels = [example['label'] for example in train_dataset]

    string_lengths = [len(text) for text in texts]

    lengths_by_class = {0: [], 1: []}
    unique_labels = [0, 1]
    display_labels = ['Negative', 'Positive']

    for i, label in enumerate(labels):
        lengths_by_class[label].append(string_lengths[i])
    
    min_len = min(string_lengths)
    max_len = max(string_lengths)
    bin_width = 350
    bins = np.arange(min_len, max_len + bin_width, bin_width)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

    axes[0].hist(lengths_by_class[0], bins=bins, edgecolor='black', color='orange')
    axes[0].set_title(f'{display_labels[0]} Class')
    axes[0].set_xlabel('String Length')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', linestyle='--')

    axes[1].hist(lengths_by_class[1], bins=bins, edgecolor='black', color='dodgerblue')
    axes[1].set_title(f'{display_labels[1]} Class')
    axes[1].set_xlabel('String Length')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', linestyle='--')

    fig.suptitle('Distribution of Review Lengths by Class', fontsize=16)

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'./plots/{filename}')


if __name__ == '__main__':
    print('Loading dataset... (this may take a while)')
    train_dataset = load_dataset("stanfordnlp/imdb", split='train')
    print(f'{colorama.Fore.GREEN}Loaded {len(train_dataset)} datapoints!{colorama.Style.RESET_ALL}')

    print('Plotting box plot... ', end='')
    plot_box_plot(train_dataset)
    print(f'{colorama.Fore.GREEN}Finished!{colorama.Style.RESET_ALL}')

    print('Plotting histogram... ', end='')
    plot_histogram(train_dataset)
    print(f'{colorama.Fore.GREEN}Finished!{colorama.Style.RESET_ALL}')