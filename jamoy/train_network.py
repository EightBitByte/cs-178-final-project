# train_network.py
#
# Trains a neural network on the training IMdB dataset.
import torch
import os
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
EMBEDDING_DIM = 128  # Dimension for token embeddings
HIDDEN_SIZE = 50     # Hidden size for the fully connected layer

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, pad_idx):
        super(SimpleNeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
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
        out = self.fc2(out)
        return out

def train_model():
    device = t_device("cuda" if cuda_is_available() else "cpu")
    print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Using device: {device}{color.Style.RESET_ALL}')

    # Load dataset
    print('Loading dataset...')
    train_dataset = load_dataset("stanfordnlp/imdb", split='train')
    print(f'{color.Fore.GREEN}Loaded {len(train_dataset)} datapoints!{color.Style.RESET_ALL}')

    # Tokenize the text
    print('Tokenizing text...')
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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    tokenized_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=['text']
    )

    tokenized_dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
    )
    print(f'{color.Fore.GREEN}Tokenizing complete!{color.Style.RESET_ALL}')

    # Load dataset into batches for training
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Begin training
    print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Starting training...{color.Style.RESET_ALL}')
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
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

        train_loss /= len(data_loader.dataset) # Average training loss for the epoch

        print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}{color.Style.RESET_ALL}')
    print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Training finished!{color.Style.RESET_ALL}')
    
    # Save the trained model
    os.makedirs('model', exist_ok=True)
    model_path = 'model/simple_imdb_classifier.pth'
    torch.save(model.state_dict(), model_path)
    print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Model saved to {model_path}!{color.Style.RESET_ALL}')

if __name__ == '__main__':
    train_model()