# train_network.py
#
# Trains a neural network on the training IMdB dataset.
import os
import torch
from torch import device as t_device
from sys import argv
from os import makedirs
from torch.cuda import is_available as cuda_is_available
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import colorama as color
from transformers import AutoTokenizer

NUM_EPOCHS = 10      # The number of epochs for training loop
EMBEDDING_DIM = 256  # Dimension for token embeddings
HIDDEN_SIZE = 50     # Hidden size for the fully connected layer
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
STATS_CSV_FILENAME = "training_stats.csv"

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, pad_idx):
        super(SimpleNeuralNetwork, self).__init__()
        self.dropout_rate = 0.5 
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  

        expanded_attention_mask = attention_mask.unsqueeze(-1).expand_as(embedded)
        
        # Zero out embeddings of padding tokens
        masked_embedded = embedded * expanded_attention_mask
        
        # Sum embeddings for non-padding tokens along the sequence length dimension
        summed_embeddings = masked_embedded.sum(dim=1) 
        
        # Count non-padding tokens for each sequence in the batch
        num_non_padding_tokens = attention_mask.sum(dim=1).unsqueeze(-1).float()
        
        # Avoid division by zero if a sequence is all padding (should not happen with proper data/truncation)
        num_non_padding_tokens = torch.clamp(num_non_padding_tokens, min=1e-9)

        pooled_embeddings = summed_embeddings / num_non_padding_tokens 

        out = self.fc1(pooled_embeddings)
        out = self.relu(out)
        out = self.dropout(out) 
        out = self.fc2(out)
        return out

def evaluate_model_during_training(model, data_loader, device, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy


def train_model(filename: str):
    device = t_device("cuda" if cuda_is_available() else "cpu")
    print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Using device: {device}{color.Style.RESET_ALL}')

    # Load dataset
    print('Loading dataset...')
    full_train_dataset = load_dataset("stanfordnlp/imdb", split='train')
    print(f'{color.Fore.GREEN}Loaded {len(full_train_dataset)} total datapoints for training/validation!{color.Style.RESET_ALL}')

    # Split dataset into training and validation sets
    split_datasets = full_train_dataset.train_test_split(test_size=0.25, shuffle=True, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test'] 
    print(f'{color.Fore.GREEN}Split dataset: {len(train_dataset)} training, {len(val_dataset)} validation datapoints.{color.Style.RESET_ALL}')

    # Tokenize the text
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = SimpleNeuralNetwork(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=2,
        pad_idx=pad_token_id
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    print('Tokenizing training data...')
    tokenized_train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text']
    )
    tokenized_train_dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
    )
    print(f'{color.Fore.GREEN}Training data tokenizing complete!{color.Style.RESET_ALL}')

    print('Tokenizing validation data...')
    tokenized_val_dataset = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text']
    )
    tokenized_val_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
    )
    print(f'{color.Fore.GREEN}Validation data tokenizing complete!{color.Style.RESET_ALL}')

    # Create DataLoaders
    train_loader = DataLoader(
        tokenized_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        tokenized_val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Begin training
    print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Starting training of "{filename}"...{color.Style.RESET_ALL}')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Extract labels from the batch
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad() # Zero the gradients before backpropagation
            loss.backward()       # Compute gradients
            optimizer.step()      # Update model parameters
            
            train_loss += loss.item() * labels.size(0) # Use batch size from labels or model_inputs

        avg_train_loss = train_loss / len(train_loader.dataset) 

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_model_during_training(model, val_loader, device, criterion)

        print(f'{color.Fore.GREEN}Epoch [{epoch+1}/{NUM_EPOCHS}]:{color.Style.RESET_ALL} '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}%')
    print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Training finished!{color.Style.RESET_ALL}')


    # Evaluate on training, validation set one last time
    val_loss, val_accuracy = evaluate_model_during_training(model, val_loader, device, criterion)
    train_loss, train_accuracy = evaluate_model_during_training(model, train_loader, device, criterion)

    print(f'{color.Fore.GREEN}Final Losses:{color.Style.RESET_ALL} '
          f'Train Loss: {avg_train_loss:.4f} | '
          f'Train Acc: {train_accuracy*100:.2f}% | '
          f'Val Loss: {val_loss:.4f} | '
          f'Val Acc: {val_accuracy*100:.2f}%')

    # Check if file exists to write header only once
    file_exists = os.path.isfile(STATS_CSV_FILENAME)

    with open(STATS_CSV_FILENAME, 'a') as file: # Open in append mode
        if not file_exists:
            file.write('model_name,train_loss,train_acc,val_loss,val_acc\n')
        model_name_for_stats = filename[:-4] if filename.endswith('.pth') else filename
        file.write(f'{model_name_for_stats},{avg_train_loss:.4f},{train_accuracy*100:.2f},{val_loss:.4f},{val_accuracy*100:.2f}\n')
    print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Stats appended to {STATS_CSV_FILENAME}!{color.Style.RESET_ALL}')
    
    # Save the trained model
    makedirs('models', exist_ok=True)
    model_path = f'models/{filename}'
    torch.save(model.state_dict(), model_path)
    print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Model saved to {model_path}!{color.Style.RESET_ALL}')

if __name__ == '__main__':
    if len(argv) < 2:
        print('Usage: train_network <model_name>')
    else:
        train_model(argv[1])