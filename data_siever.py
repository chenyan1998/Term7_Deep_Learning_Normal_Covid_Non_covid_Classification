import torch





def data_siever(input_images, input_labels, dataloader):
    input_labels_list = []
    input_iamge_list = []
    another_labels_list = []
    another_iamge_list = []
    cov_non_dict = {}
    input_dict = {}
    i = -1
    for images, labels in dataloader:
        i += 1
        images = images.tolist()
        labels = labels.tolist()
        for idx in range(len(images)):
            key = images[idx][0][0][67]
            value = labels[idx][1]
            cov_non_dict[key] = value

    for idx in range(len(input_labels)):

        key = input_images[idx][0][0][67]
        value = input_labels[idx]
        input_dict[key] = value
    
    i = -1
    for key in input_dict:
        i += 1
        try:
            label = cov_non_dict[key]
            input_iamge_list.append(input_images[i])
            input_labels_list.append(label) 

        except:
            another_iamge_list.append(input_images[i])
            another_labels_list.append(0)
            

    return input_labels_list, input_iamge_list, another_labels_list, another_iamge_list 
