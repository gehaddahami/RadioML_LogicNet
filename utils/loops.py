# Imports
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 
from sklearn.metrics import accuracy_score
import torch


# TODO: after making sure the functions are properly working, remove all extra priniting statements 
def train_loop(model, train_loader, optimizer, criterion, options):
    losses = []
    model.train()

    for (inputs, labels, snr) in tqdm(train_loader):
        print('break')
        if options['cuda']: 
            inputs, labels = inputs.cuda(), labels.cuda()

        # Forward pass 
        output = model(inputs) 
        loss = criterion(output, labels)

        # Backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Use .item() to get the scalar value of the loss

    return losses
    

def test_loop(model, test_loader, options):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        
        for (inputs, labels, snr) in tqdm(test_loader):

            if options['cuda']: 
                inputs, labels = inputs.cuda(), labels.cuda()
    
            outputs = model(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)

            y_true.append(labels.numpy())  # Move labels back to CPU for concatenation
            y_pred.append(pred.reshape(-1).numpy())

    # printing the accuracy of the model (2nd method)
    y_true = np.concatenate(y_true)
    print(y_true.shape)
    y_pred = np.concatenate(y_pred)
    print(y_pred.shape)

    return accuracy_score(y_true, y_pred)



# plotting losses and/or accuracy of the model
def display_loss(losses, title = 'Training loss', xlabel= 'Iterations', ylabel= 'Loss'):
    x_axis = [i for i in range(len(losses))] 
    plt.plot(x_axis, losses)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)