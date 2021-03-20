import torch 

def accuracy_per_batch(labels_per_batch, pred_per_batch):
    equality = labels_per_batch[:] == pred_per_batch
    acc_per_batch = equality.type(torch.FloatTensor).mean()

    return acc_per_batch

def recall_per_batch(labels_per_batch, pred_per_batch): 
    actual_positive = labels_per_batch[:].sum()
    print('actual_positve', actual_positive)

    true_positive = 0 
    for i in range(len(labels_per_batch)):
        if labels_per_batch[i] == 1.0:
            if pred_per_batch[i] == labels_per_batch[i]:
                true_positive += 1

    print('true_positive', true_positive)
    recall_per_batch = true_positive / actual_positive
    # f1_score_per_batch = 2 * precision* recall / (precision + recall)

    return recall_per_batch

def precision_per_batch(labels_per_batch, pred_per_batch): 
    predicted_positive = pred_per_batch.sum()
    print('predicted_positive', predicted_positive)

    true_positive = 0 
    for i in range(len(pred_per_batch)):
        if pred_per_batch[i] == 1.0:
            if labels_per_batch[i] == pred_per_batch[i]:
                true_positive += 1

    print('true_positive', true_positive)
    precision_per_batch = true_positive / predicted_positive

    return precision_per_batch