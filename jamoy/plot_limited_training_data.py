# plot_data.py
#
# Provides a few functions for plotting data.

import csv
import matplotlib.pyplot as plt

testing_file: str = "training_stats.csv"
fp_total = 1494
fp_percentage = 11.95
fn_total = 1955
fn_percentage = 15.64

def plot_percentages():
    """Plots the validation accuracy over the amount of training data available."""
    percentages: list[int] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    with open(testing_file, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if row[0].endswith('pct'):
                percentage = int(row[0][-5:-3])
                train_acc = float(row[2])
                val_acc = float(row[4])

                percentages.append(percentage)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
        
        plt.clf()
        plt.plot(percentages, train_accs, label='Training Accuracy', marker='o')
        plt.plot(percentages, val_accs, label='Validation Accuracy', marker='o')
        plt.title('Learning Curves for Neural Network (model 5)')
        plt.xlabel('% of training data available')
        plt.ylabel('% of accuracy')
        plt.legend()
        plt.savefig('./plots/learning-curves.png')

def plot_fp_fn():
    """Plots the false positives and false negatives in a bar plot."""
    plt.clf()
    plt.title('Number of False Positive and False Negative\nInstances in Neural Network (model 5)') 
    plt.ylabel('Number of Testing Examples')
    plt.bar(["False Positive", "False Negative"], [fp_total, fn_total], color=['dodgerblue', 'darkorange'])
    plt.savefig('./plots/fp-fn.png')


if __name__ == '__main__':
    plot_percentages()    
    plot_fp_fn()