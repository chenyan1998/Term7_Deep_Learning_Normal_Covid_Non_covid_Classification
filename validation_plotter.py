from norm_infected_model import norm_infected_model
from test import predict_val
from dataloader_covid_non import get_data_obj_covid 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


model = norm_infected_model()
bs_val = 8 #Change according to dataset
ld_train_covid, ld_test_covid, ld_val_covid= get_data_obj_covid()
val_loader = DataLoader(ld_val_covid, batch_size = bs_val, shuffle = True)

all_pred, all_labels, all_images, val_loss, acc, recall, precision = predict_val(val_loader, model, './checkpoints/model1_5/epoch-14.pt', 'cpu')

def validation_plotter(all_pred, all_labels, all_images, acc):
    #Graph Plotting
    # settings
    h, w = 10, 10        # for raster image
    nrows, ncols = 4, 4  # array of sub-plots
    figsize = [6, 8]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        arr = all_images[i]
        img = np.squeeze(arr)
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Label:"+str(int(all_labels[i][1]))+", Pred:"+str(all_pred[i]))
        print(all_labels[i])
        print(all_pred[i])
    plt.suptitle("Accuracy of Validation Set: "+str(acc.item()))
    plt.tight_layout(True)
    plt.show()

validation_plotter(all_pred, all_labels, all_images, acc)