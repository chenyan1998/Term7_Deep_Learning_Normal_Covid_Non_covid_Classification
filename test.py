import torch 
from train import load_model 
from torch import nn
from metrics import accuracy_per_batch, recall_per_batch, precision_per_batch

def predict_model1(testloader, model, path, device):
    model = load_model(model, path)
    model.eval()

    test_loss = 0
    accuracy = 0
    recall = 0 
    precision = 0 

    criterion = nn.BCELoss()
    all_pred = []
    all_labels = []
    all_images = []

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        label = labels[:,1].int().tolist()
        all_labels += label
        all_images += images.tolist()
                
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
    

    return all_pred, all_labels, all_images, test_loss, accuracy, recall, precision 

def predict_model2(labels, images, model, path, device):
    model = load_model(model, path)
    model.eval()

    test_loss = 0
    accuracy = 0
    recall = 0 
    precision = 0 

    criterion = nn.BCELoss()
    all_pred = []
    all_labels = []
    all_images = []
    images_tensor = []
    label_batch = []

    for i in range(len(labels)):
        label = labels[i]
        image = images[i]

        all_labels.append(label)

        label = torch.Tensor([label])
        label_batch.append(label)

        all_images += image
        image = torch.Tensor(image)
        images_tensor.append(image)

        while ((((i+1) % 8) == 0) or (i == 25)):
            
            images_tensor = torch.stack(images_tensor)
            print(images_tensor.size())
            label_batch = torch.stack(label_batch)
            label_batch = torch.flatten(label_batch)
            images_tensor = images_tensor.to(device)
            output = model.forward(images_tensor)
            output = torch.flatten(output)

            test_loss += criterion(output, label_batch).item()
            pred = torch.flatten(torch.round(output)).int()
            all_pred += pred.tolist()
            accuracy += accuracy_per_batch(label_batch, pred)
            recall += recall_per_batch(label_batch, pred)
            precision += precision_per_batch(label_batch, pred)
            images_tensor = []
            label_batch = []
            break

    test_loss = test_loss/4
    accuracy = accuracy/4
    recall = recall/4
    precision = precision/4

    return all_pred, all_labels, all_images, test_loss, accuracy, recall, precision 

def input_from_model1(images, labels, preds):

    images_ls = []
    labels_ls = []

    for i in range(len(preds)):
        if preds[i] == 1:
            images_ls.append(images[i])
            labels_ls.append(labels[i])
    
    return images_ls, labels_ls # Need find a way to load in batches like dataloader for line 18 to work 
                
        



