from norm_infected_model import norm_infected_model
from test import predict, predict_test
from dataloader_norm_infected import get_data_obj 
from torch.utils.data import DataLoader


model = norm_infected_model()

bs_test = 64

ld_train, ld_test, ld_val= get_data_obj()
test_loader = DataLoader(ld_test, batch_size = bs_test, shuffle = True)
all_pred, all_labels, all_images, test_loss, acc, recall, precision = predict(test_loader, model, './checkpoints/model1_13/epoch-14.pt', 'cpu')
all_pred, all_labels, all_images, test_loss, acc, recall, precision = predict_test(all_pred, all_images, model, './checkpoints/model1_13/epoch-14.pt', 'cpu')

print('all_pred', all_pred)
print('all_labels', all_labels)
print('test_loss', test_loss)
print('acc', acc)
print('recall', recall)
print('precision', precision)

