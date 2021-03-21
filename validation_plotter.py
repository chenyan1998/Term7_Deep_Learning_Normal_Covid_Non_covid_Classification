from norm_infected_model import norm_infected_model
from test import predict_model1, predict_model2
from dataloader_covid_non import get_data_obj_covid 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# model = norm_infected_model()
# bs_val = 8 #Change according to dataset
# ld_train_covid, ld_test_covid, ld_val_covid= get_data_obj_covid()
# val_loader = DataLoader(ld_val_covid, batch_size = bs_val, shuffle = True)

# all_pred, all_labels, all_images, val_loss, acc, recall, precision = predict_model1(val_loader, model, './checkpoints/model1_13/epoch-14.pt', 'cpu')

def validation_plotter(all_pred, all_labels, all_images, acc, acc2, norm_images, norm_labels, other_images, other_labels):
    #Graph Plotting
    # settings
    h, w = 10, 10        # for raster image
    nrows, ncols = 6, 4  # array of sub-plots
    figsize = [6, 8]     # figure size, inches
    combined_images = all_images + norm_images + other_images
    combined_labels = all_labels + norm_labels + other_labels
    differentiator = len(all_labels)
    print(len(combined_images))

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        
        # write row/col indices as axes' title for identification
        if i < differentiator:
            arr = combined_images[i]
            img = np.squeeze(arr)
            axi.imshow(img)
            # get indices of row/column
            rowid = i // ncols
            colid = i % ncols
            axi.set_title("Label:"+str(int(combined_labels[i]))+", Pred:"+str(all_pred[i]))
        elif differentiator <= i < len(combined_labels):
            arr = combined_images[i]
            img = np.squeeze(arr)
            axi.imshow(img)
            # get indices of row/column
            rowid = i // ncols
            colid = i % ncols
            axi.set_title("Label:"+str(int(combined_labels[i]))+", Pred:"+str(0))
        else:
            continue
        
    plt.suptitle("Accuracy of Validation Set: "+str((acc.item()+acc2.item())/2))
    plt.tight_layout(True)
    plt.show()

# validation_plotter(all_pred, all_labels, all_images, acc)