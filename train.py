import torch
from torch import nn
from torch import optim
import time 
import os 
from norm_infected_model import norm_infected_model
import numpy as np 
from metrics import accuracy_per_batch, precision_per_batch, recall_per_batch
import matplotlib.pyplot as plt

# Define validation function 
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    recall_score = 0
    precision_score = 0 
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        labels = labels[:,1]
        output = torch.flatten(output)
        test_loss += criterion(output, labels).item()

        pred = torch.flatten(torch.round(output)).int()
        accuracy += accuracy_per_batch(labels, pred)

        recall_score += recall_per_batch(labels, pred)

        precision_score += precision_per_batch(labels, pred)

    return test_loss, accuracy, recall_score, precision_score 

def train(model, n_epoch, lr, device, trainloader, validloader, model_dir):

    # Define criterion and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    # optimizer = optim.RMSprop(model.parameters(), lr = lr)
    # optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0 
    running_loss = 0
    train_accuracy = 0
    train_recall_score = 0
    train_precision_score = 0 
    print_every = 1

    train_loss_ls = []
    train_acc_ls = []
    train_recall_score_ls = []
    train_precision_score_ls = []

    val_loss_ls = []
    val_acc_ls = []
    val_recall_score_ls = []
    val_precision_score_ls = []

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            pred = torch.flatten(torch.round(output)).int()

            labels = labels[:,1]
            output = torch.flatten(output)
            #print("output", output)
            #print("label", labels)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_accuracy += accuracy_per_batch(labels, pred)

            train_recall_score += recall_per_batch(labels, pred)

            train_precision_score += precision_per_batch(labels, pred)

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    val_loss, val_accuracy, val_recall_score, val_precision_score = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} - ".format(val_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(val_accuracy/len(validloader)),
                      "Validation Recall-score: {:.3f}".format(val_recall_score/len(validloader)),
                      "Validation Precision-score: {:.3f}".format(val_precision_score/len(validloader))
                )

                train_loss_ls.append(running_loss/print_every)
                train_acc_ls.append(train_accuracy/print_every)
                train_recall_score_ls.append(train_recall_score/print_every)
                train_precision_score_ls.append(train_precision_score/print_every)

                val_loss_ls.append(val_loss/len(validloader))
                val_acc_ls.append(val_accuracy/len(validloader))
                val_recall_score_ls.append(val_recall_score/len(validloader))
                val_precision_score_ls.append(val_precision_score/len(validloader))

                running_loss = 0
                train_accuracy = 0 
                train_recall_score = 0 
                train_precision_score = 0 

                # Make sure training is back on
                model.train()

        if (e % 10 == 0) or (e == epochs-1): 
            save_checkpoint(model, optimizer, epochs, train_loss_ls, val_loss_ls, os.path.join(model_dir, 'epoch-{}.pt'.format(e)))

    print(f"Run time: {(time.time() - start)/60:.3f} min")

    plt.plot(train_loss_ls, label="train_loss")
    plt.plot(val_loss_ls, label="val loss")

    plt.plot(train_acc_ls, label="train_acc")
    plt.plot(val_acc_ls, label="val_acc")

    plt.plot(train_recall_score_ls, label="train_recall_score")
    plt.plot(val_recall_score_ls, label="val_recall_score")

    plt.plot(train_precision_score_ls, label="train_precision_score")
    plt.plot(val_precision_score_ls, label="val_precision_score")
    
    plt.legend()
    plt.xlabel("training step")
    plt.show()

    return model

# Define function to save checkpoint
def save_checkpoint(model, optimizer, n_epoch, train_loss_ls, val_loss_ls, path):
    checkpoint = {'state_dict': model.state_dict(),
                  'opti_state_dict': optimizer.state_dict(),
                  'train_loss_ls': train_loss_ls, 
                  'valid_loss_ls': val_loss_ls
                  }
    torch.save(checkpoint, path)


# Define function to load model
def load_model(model, path):
    cp = torch.load(path)
    
    # Add model info 
    model.optimizer_state_dict = cp['opti_state_dict']
    model.load_state_dict(cp['state_dict'])

    return model 
