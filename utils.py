
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import configs  
import time
import os
import torch.nn.functional as F


def imbalanced_class_weights(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight    

def load_split_train_test(data_dir,batch_size = 64):
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                                           [0.229, 0.224, 0.225]) # case I didn't get good results
                                       ])

    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                      ])

    train_data = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=train_transforms)
    val_data = datasets.ImageFolder(os.path.join(data_dir,"val"), transform=test_transforms)

    print(f"Total Train images :",len(train_data.imgs))
    # For unbalanced dataset we create a weighted sampler                       
    train_weights = imbalanced_class_weights(train_data.imgs, len(train_data.classes))                                                                
    train_weights = torch.DoubleTensor(train_weights)                                       
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))                     

    print(f"Total Test images :",len(val_data.imgs))
    val_weights = imbalanced_class_weights(val_data.imgs, len(val_data.classes))                                                                
    val_weights = torch.DoubleTensor(val_weights)                                       
    val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_weights, len(val_weights))                     
                                                           
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = False,                              
                                sampler = train_sampler, num_workers=4, pin_memory=True) 
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle = False,                              
                                sampler = val_sampler, num_workers=4, pin_memory=True) 



    # labels = test_data.imgs
    # print("here :",len(labels[:]))
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(valid_size * num_train))
    # np.random.shuffle(indices)
    # # from torch.utils.data.sampler import SubsetRandomSampler
    # # train_idx, test_idx = indices[split:], indices[:split]
    # # train_sampler = SubsetRandomSampler(train_idx)
    # # test_sampler = SubsetRandomSampler(test_idx)

    

    # weights = 1 / torch.Tensor(class_sample_count)
    # weights = weights.double()
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
    # trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_loader, val_loader



# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            configs.CLASSES[preds[idx]],
            probs[idx] * 100.0,
            configs.CLASSES[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        npimg = npimg.astype(int)
        plt.imshow(np.transpose(np.clip(npimg,0,255), (1, 2, 0)))


def save_checkpoint(path,epoch,model,optimizer,loss):
    
    os.makedirs(path,exist_ok=True)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': loss,
                }, os.path.join(path,f"{time.time()}.tar"))