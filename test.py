import torch 
from train import load_model 
from torch import nn
from metrics import accuracy_per_batch

def predict(testloader, path, device):
    model = load_model(path)
    model.eval()

    test_loss = 0
    accuracy = 0
    criterion = nn.crossEntropyLoss()
    all_pred = []

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        pred = torch.flatten(torch.round(output)).int()
        all_pred += pred.tolist()
        accuracy += accuracy_per_batch(labels, pred)

    return all_pred, test_loss, accuracy



