import torch 
from train import load_model 
from torch import nn
from metrics import accuracy_per_batch, recall_per_batch, precision_per_batch

def predict(testloader, model, path, device):
    model = load_model(model, path)
    model.eval()

    test_loss = 0
    accuracy = 0
    recall = 0 
    precision = 0 

    criterion = nn.BCELoss()
    all_pred = []

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)

        labels = labels[:,1]
        output = torch.flatten(output)
        test_loss += criterion(output, labels).item()
        
        pred = torch.flatten(torch.round(output)).int()
        all_pred += pred.tolist()
        accuracy += accuracy_per_batch(labels, pred)

        recall += recall_per_batch(labels, pred)

        precision += precision_per_batch(labels, pred)

    test_loss = test_loss/len(testloader)
    accuracy = accuracy/len(testloader)
    recall = recall/len(testloader)
    precision = precision/len(testloader)

    return all_pred, test_loss, accuracy, recall, precision 

    def input_from_model1(images, labels, preds):

        images_ls = []
        labels_ls = []

        for i in range(len(preds)):
            if pred[i] == 1:
                images_ls.append(images[i])
                labels_ls.append(labels[i])
        
        return images_ls, labels_ls # Need find a way to load in batches like dataloader for line 18 to work 
                
        



