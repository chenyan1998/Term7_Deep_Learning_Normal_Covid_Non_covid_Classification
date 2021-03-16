import torch
from torch import nn
from torch import optim
import time 
import os 
from norm_infected_model import norm_infected_model
import numpy as np 
from metrics import accuracy_per_batch, f1_per_batch
import matplotlib.pyplot as plt

# Define validation function 
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    f1_score = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        pred = torch.flatten(torch.round(output)).int()
        accuracy += accuracy_per_batch(labels, pred)

        f1_score += f1_per_batch(labels, pred)

    return test_loss, accuracy, f1_score

def train(model, n_epoch, lr, device, trainloader, validloader, model_dir):

    # Define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0 
    running_loss = 0
    train_accuracy = 0
    train_f1_score = 0
    print_every = 1

    train_loss_ls = []
    train_acc_ls = []
    train_f1_score_ls = []
    val_loss_ls = []
    val_acc_ls = []
    val_f1_score_ls = []

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            pred = torch.flatten(torch.round(output)).int()

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_accuracy += accuracy_per_batch(labels, pred)

            train_f1_score += f1_per_batch(labels, pred)

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    val_loss, val_accuracy, val_f1_score = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Training Loss: {:.3f} - ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} - ".format(val_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(val_accuracy/len(validloader)),
                      "Validation F1-score: {:.3f}".format(val_f1_score/len(validloader))
                )

                train_loss_ls.append(running_loss/print_every)
                train_acc_ls.append(train_accuracy/print_every)
                train_f1_score_ls.append(train_f1_score/print_every)

                val_loss_ls.append(val_loss/len(validloader))
                val_acc_ls.append(val_accuracy/len(validloader))
                val_f1_score_ls.append(val_f1_score/len(validloader))

                running_loss = 0
                accuracy = 0 
                f1_score = 0 

                # Make sure training is back on
                model.train()

        if (e % 10 == 0): 
            save_checkpoint(model, optimizer, train_loss_ls, val_loss_ls, os.path.join(model_dir, 'epoch-{}.pt'.format(e)))

    print(f"Run time: {(time.time() - start)/60:.3f} min")

    plt.plot(train_loss_ls, label="train_loss")
    plt.plot(val_loss_ls, label="val loss")
    plt.plot(train_acc_ls, label="train_acc")
    plt.plot(val_acc_ls, label="val_acc")
    plt.plot(train_f1_score_ls, label="train_f1_score")
    plt.plot(val_f1_score_ls, label="val_f1_score")
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
def load_model(path):
    cp = torch.load(path)
    
    # Make classifier
    model = norm_infected_model() 
    # Add model info 
    model.optimizer_state_dict = cp['opti_state_dict']
    model.load_state_dict(cp['state_dict'])

    return model 
