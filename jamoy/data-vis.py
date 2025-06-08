# data-vis.py
# 
# Used to import and visualize the dataset.
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

def plot_box_plot(plot_data, display_labels, filename='box_plot.png'):
    plt.boxplot(plot_data, labels=display_labels, patch_artist=True)

    plt.title('Distribution of String Lengths by Class (Box Plot)')
    plt.xlabel('Class Label')
    plt.ylabel('String Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'./plots/{filename}')


if __name__ == '__main__':
    print('Loading dataset.. (this may take a while)')
    train_dataset = load_dataset("stanfordnlp/imdb", split='train')
    print(f'Loaded {len(train_dataset)} datapoints!')

    # Plot length of string against class
    texts = [example['text'] for example in train_dataset]
    labels = [example['label'] for example in train_dataset]

    string_lengths = [len(text) for text in texts]

    lengths_by_class = {0: [], 1: []}
    unique_labels = [0, 1]
    display_labels = ['Negative', 'Positive']

    for i, label in enumerate(labels):
        lengths_by_class[label].append(string_lengths[i])
    
    plot_data = [lengths_by_class[label] for label in unique_labels]

    plot_box_plot(plot_data, display_labels)