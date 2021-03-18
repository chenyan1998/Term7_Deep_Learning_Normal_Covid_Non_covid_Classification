import torch 

def accuracy_per_batch(labels_per_batch, pred_per_batch):
    equality = labels_per_batch[:] == pred_per_batch
    acc_per_batch = equality.type(torch.FloatTensor).mean()

    return acc_per_batch

def f1_per_batch(labels_per_batch, pred_per_batch): 
    target_true = labels_per_batch[:].sum()
    predicted_true = pred_per_batch.sum()
    equality = labels_per_batch[:] == pred_per_batch
    correct_true = equality.type(torch.FloatTensor).sum()
    recall = correct_true / target_true
    precision = correct_true / predicted_true
    f1_score_per_batch = 2 * precision* recall / (precision + recall)

    return f1_score_per_batch