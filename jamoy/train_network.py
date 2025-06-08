# train_network.py
#
# Trains a neural network on the training IMdB dataset.
from torch import device as t_device
from torch.cuda import is_available as cuda_is_available
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import colorama as color
from transformers import AutoTokenizer

# The number of epochs for training loop.
NUM_EPOCHS = 10

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

def train_model():
    device = t_device("cuda" if cuda_is_available() else "cpu")
    print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Using device: {device}{color.Style.RESET_ALL}')

    model = SimpleNeuralNetwork(input_size=12, hidden_size=50, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load and process dataset
    print('Loading dataset...')
    train_dataset = load_dataset("stanfordnlp/imdb", split='train')
    print(f'{color.Fore.GREEN}Loaded {len(train_dataset)} datapoints!{color.Style.RESET_ALL}')

    print('Tokenizing...')
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True
        )

    tokenized_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text']
    )

    print(tokenized_dataset.column_names)

    tokenized_dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
    )

    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    print("\nFirst batch from DataLoader:")
    for batch_idx, batch in enumerate(data_loader):
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  Batch token_type_ids shape: {batch['token_type_ids'].shape}")
        print(f"  Batch labels shape: {batch['label'].shape}")

        if batch_idx == 0:
            break

    # Begin training
    # # print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Starting training...{color.Style.RESET_ALL}')
    # for epoch in range(NUM_EPOCHS):
    #     model.train()
    #     train_loss = 0.0
    #     for batch_idx, (inputs, labels) in enumerate(train_loader):
    #             inputs, labels = inputs.to(device), labels.to(device) # Move data to device

    #             # Forward pass
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)

    #             # Backward pass and optimization
    #             optimizer.zero_grad() # Zero the gradients before backpropagation
    #             loss.backward()       # Compute gradients
    #             optimizer.step()      # Update model parameters

    #             train_loss += loss.item() * inputs.size(0)

    #         train_loss /= len(train_loader.dataset) # Average training loss for the epoch




if __name__ == '__main__':
    train_model()