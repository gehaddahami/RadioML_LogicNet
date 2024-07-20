import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Importing the model and the data loader
from data.data_loading import Radioml_18
from model.model import QuantizedRadiomlNEQ
from utils.loops import train_loop, test_loop, display_loss


options = {
    'cuda' : None,
    'log_dir' : None, 
    'checkpoint' : None 
    }
	

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate QuantizedRadioml model")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for data loaders')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--t0', type=int, default=5, help='T_0 parameter for CosineAnnealingWarmRestarts scheduler')
    parser.add_argument('--t_mult', type=int, default=1, help='T_mult parameter for CosineAnnealingWarmRestarts scheduler')
    return parser.parse_args()



def load_data(dataset_path, batch_size):
    dataset = Radioml_18(dataset_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.validation_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
    return train_loader, validation_loader, test_loader


def create_model():
    model_config = {
        "input_length": 2,
        "hidden_layers": [64] * 7 + [128] * 2,
        "output_length": 24,
        "input_bitwidth": 8,
        "hidden_bitwidth": 8,
        "output_bitwidth": 8,
        "input_fanin": 3,
        "conv_fanin": 2,
        "hidden_fanin": 50,
        "output_fanin": 128
    }
    model = QuantizedRadiomlNEQ(model_config=model_config)
    return model


def main():
    args = parse_args()

    # Load data
    train_loader, validation_loader, test_loader = load_data(args.dataset_path, args.batch_size)

    # Create model
    model = create_model()
    print(model)
    
    # load from checkpoint if available: 
    if options['checkpoint'] is not None:
        print(f'Loading pre-trained checkpoint {options["checkpoint"]}')
        checkpoint = torch.load(options['checkpoint'], map_location = 'cpu')
        model.load_state_dict(checkpoint['model_dict'])
        print(f'Checkpoint loaded successfully')
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)

    # Training loop
    running_loss = []
    accuracy = []
    for epoch in range(args.num_epochs):
        loss_epoch = train_loop(model, train_loader, optimizer, criterion)
        test_acc, predictions, labels = test_loop(model, validation_loader)
        print(f"Epoch {epoch}: Training loss = {np.mean(loss_epoch):.6f}, validation accuracy = {test_acc:.6f}")
        running_loss.append(loss_epoch)
        accuracy.append(test_acc)

        # Step the scheduler
        scheduler.step()

    # Optionally, plot the running loss and accuracy
    display_loss(running_loss)


if __name__ == "__main__":
    main()
