import argparse
import os 
from train import save_checkpoint, train 
from norm_infected_model import norm_infected_model
from covid_non_model import covid_non_model
from dataloader_norm_infected import get_data_obj
from torch.utils.data import DataLoader
from dataloader_covid_non import get_data_obj_covid 
from test import predict_model1, predict_model2, input_from_model1


parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--epochs", type=int, default=10, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir_norm_inf_model", help="save model")
parser.add_argument("--save_dir_covid_non_model", help="save model")

args = parser.parse_args()

# train normal, infected model 
# ld_train, ld_test, ld_val= get_data_obj()

# bs_train = 64 
# bs_test = 32
# bs_val = 3

# train_loader = DataLoader(ld_train, batch_size = bs_train, shuffle = True)
# test_loader = DataLoader(ld_test, batch_size = bs_test, shuffle = True)
# val_loader = DataLoader(ld_val, batch_size = bs_val , shuffle = True)

# model1 = norm_infected_model()
# model1 = train(model1, args.epochs, args.learning_rate, args.gpu, train_loader, test_loader, args.save_dir_norm_inf_model)

# train covid, non_covid model 
# ld_train_covid, ld_test_covid, ld_val_covid= get_data_obj_covid()

# train_loader_covid = DataLoader(ld_train_covid, batch_size = bs_train, shuffle = True)
# test_loader_covid = DataLoader(ld_test_covid, batch_size = bs_test, shuffle = True)
# val_loader_covid = DataLoader(ld_val_covid, batch_size = bs_val, shuffle = True)

# model2 = covid_non_model()
# model2 = train(model2, args.epochs, args.learning_rate, args.gpu, train_loader_covid, test_loader_covid, args.save_dir_covid_non_model)

# Predict using the whole model 
model1 = norm_infected_model()
path1 = './checkpoints/model1_13/epoch-14.pt' 
device = 'cpu'

ld_train, ld_test, ld_val= get_data_obj()

bs_val = 3

val_loader = DataLoader(ld_val, batch_size = bs_val , shuffle = True)

all_pred, all_labels, all_images, test_loss, accuracy, recall, precision = predict_model1(val_loader, model1, path1, device)

model2 = covid_non_model()
path2 = './checkpoints/model2_6/epoch-9.pt'

input_images, input_labels = input_from_model1(all_images, all_labels, all_pred)
all_pred, all_labels, all_images, test_loss, accuracy, recall, precision = predict_model2(input_labels, input_images, model2, path2, device)

print("Predictions: ", all_pred)
print("Labels: ", all_labels)
print(test_loss)
print(accuracy)
print(recall)
print(precision)

