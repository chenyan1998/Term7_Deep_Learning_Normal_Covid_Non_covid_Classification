from covid_non_model import covid_non_model
from test import predict
from dataloader_covid_non import get_data_obj_covid 
from torch.utils.data import DataLoader


model = covid_non_model()

bs_test = 64

ld_train_covid, ld_test_covid, ld_val_covid= get_data_obj_covid()
test_loader_covid = DataLoader(ld_test_covid, batch_size = bs_test, shuffle = True)

all_pred, test_loss, acc, recall, precision = predict(test_loader_covid, model, './checkpoints/epoch-29.pt', 'cpu')

print('all_pred', all_pred)
print('test_loss', test_loss)
print('acc', acc)
print('recall', recall)
print('precision', precision)

