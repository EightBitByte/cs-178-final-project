# test_network.py
#
# Tests the neural network on the IMdB training and testing dataset.
import torch
import os
from torch import device as t_device
from sys import argv
from torch.cuda import is_available as cuda_is_available
from datasets import load_dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import colorama as color
from transformers import AutoTokenizer
from train_network import SimpleNeuralNetwork

NUM_EPOCHS = 10      # The number of epochs for training loop
EMBEDDING_DIM = 128  # Dimension for token embeddings
HIDDEN_SIZE = 50     # Hidden size for the fully connected layer

def evaluate_model(model, data_loader, device, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    false_positives = 0
    false_negatives = 0
    total_actual_positives = 0
    total_actual_negatives = 0

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

            # Calculate False Positives (FP) and False Negatives (FN)
            # Assuming class 1 is positive and class 0 is negative
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            
            total_actual_positives += (labels == 1).sum().item()
            total_actual_negatives += (labels == 0).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    fp_percentage = (false_positives / total_actual_negatives * 100) if total_actual_negatives > 0 else 0
    fn_percentage = (false_negatives / total_actual_positives * 100) if total_actual_positives > 0 else 0
    return avg_loss, accuracy, false_positives, false_negatives, fp_percentage, fn_percentage


def test_model(model_path: str):
    device = t_device("cuda" if cuda_is_available() else "cpu")
    print(f'{color.Fore.BLUE}{color.Style.BRIGHT}Using device: {device}{color.Style.RESET_ALL}')

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    print(f'{color.Fore.GREEN}Tokenizer loaded!{color.Style.RESET_ALL}')

    print(f'Loading model from {model_path}...')
    model_instance = SimpleNeuralNetwork(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=2,
        pad_idx=pad_token_id
    ).to(device)
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    print(f'{color.Fore.GREEN}Model loaded and weights populated!{color.Style.RESET_ALL}')

    criterion = nn.CrossEntropyLoss()

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    datasets_to_evaluate = {}

    # Load original train dataset
    print('Loading original train dataset...')
    original_train_dataset = load_dataset("stanfordnlp/imdb", split='train')
    print(f'{color.Fore.GREEN}Loaded {len(original_train_dataset)} original training datapoints!{color.Style.RESET_ALL}')

    # Split the original train dataset into new train and validation sets (75/25)
    print('Splitting dataset into train and validation (75/25)...')
    split_datasets = original_train_dataset.train_test_split(test_size=0.25, shuffle=True, seed=42)
    datasets_to_evaluate['train'] = split_datasets['train']
    datasets_to_evaluate['validation'] = split_datasets['test'] 
    print(f'{color.Fore.GREEN}Split complete: {len(datasets_to_evaluate["train"])} training, {len(datasets_to_evaluate["validation"])} validation datapoints.{color.Style.RESET_ALL}')

    print('Loading testing dataset...')
    testing_dataset = load_dataset("stanfordnlp/imdb", split='test')
    datasets_to_evaluate['test'] = testing_dataset
    print(f'{color.Fore.GREEN}Loaded {len(original_train_dataset)} original testing datapoints!{color.Style.RESET_ALL}')

    for split_name, dataset_obj in datasets_to_evaluate.items():
        print(f'\n{color.Fore.CYAN}{color.Style.BRIGHT}Processing {split_name} dataset...{color.Style.RESET_ALL}')
        print(f'Tokenizing {split_name} dataset...')
        tokenized_data = dataset_obj.map(tokenize_function, batched=True, remove_columns=['text'])
        tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
        print(f'{color.Fore.GREEN}Tokenizing for {split_name} complete!{color.Style.RESET_ALL}')

        data_loader = DataLoader(tokenized_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        print(f'Evaluating on {split_name} dataset...')
        avg_loss, accuracy, fp, fn, fp_perc, fn_perc = evaluate_model(model_instance, data_loader, device, criterion)
        print(f'{color.Fore.GREEN}{color.Style.BRIGHT}Results for {split_name} dataset:{color.Style.RESET_ALL}')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {accuracy*100:.2f}%{color.Style.RESET_ALL}')
        print(f'  False Positives: {fp} ({fp_perc:.2f}% of actual negatives)')
        print(f'  False Negatives: {fn} ({fn_perc:.2f}% of actual positives)')

if __name__ == '__main__':
    model_file_path = os.path.join('models', argv[1])
    if not os.path.exists(model_file_path):
        print(f"{color.Fore.RED}{color.Style.BRIGHT}Error: Model file not found at {model_file_path}{color.Style.RESET_ALL}")
        print(f"Please ensure the model has been trained and saved to this location, or update the path.")
    else:
        test_model(model_file_path)