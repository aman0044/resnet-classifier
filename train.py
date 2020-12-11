#
# train.py
# @author amangupta0044@gmail.com
# @description 
# @created 2020-12-09T16:35:56.524Z+05:30
# @last-modified 2020-12-11T20:05:30.671Z+05:30
#

########### Help ###########
'''
python train.py \
    --data_dir /Users/aman.gupta/Documents/eagleview/utilities/onsite_data_fetch/fetched_images/annotated_combined_thumbnail_after_may_2020/splitted_letterbox_training_data \
    --log_dir ./logs \
    --epochs 1 \
    --save_interval 5 \
    --print_interval 1 \
    --batch_size 64 \
    --name exp0

'''
#############################


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
import os
from utils import (load_split_train_test,
                    plot_classes_preds,
                    save_checkpoint)
from torch.utils.tensorboard import SummaryWriter
import time
from model import Model
import sys
import configs



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="this script trains the classification model")
    
    parser.add_argument("--data_dir",required = True,help="training data path")
    parser.add_argument("--log_dir",required=False,default="./logs",type=str,help="dir to save logs")
    parser.add_argument("--epochs",default=10,type =int, help="number of epochs to train a model")
    parser.add_argument("--save_interval",default=100,type = int,help="interval to save model")
    parser.add_argument("--print_interval",default=10,type = int,help="interval to print log")
    parser.add_argument("--lr",default=0.003,type = float,help="learning rate")
    parser.add_argument("--batch_size",default=4,type = int,help="batch size")
    parser.add_argument("--test_split",default=0.2,type = float,help="test split out of 1.0")
    parser.add_argument("--name",default="exp0",type = str,help="experiment name")

    
    
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)

    #tensorboard writter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(args.log_dir)

    ##load data
    data_dir = args.data_dir
    trainloader, testloader = load_split_train_test(data_dir, args.batch_size)
    print(trainloader.dataset.classes)
    
    # sys.exit()

    ##load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_layers = len(configs.CLASSES)
    model_obj = Model(output_layers,device,args.lr)
    model,optimizer,criterion = model_obj.model,model_obj.optimizer,model_obj.criterion

    ## training loop
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = args.print_interval
    train_losses, test_losses = [], []
    try:
        print("Training Started")
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()
                            
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/len(trainloader))
                    test_losses.append(test_loss/len(testloader))   

                    # ...log the running loss
                    writer.add_scalar('loss/training_loss',
                                    running_loss / print_every,
                                    global_step=epoch * len(trainloader) + steps)
                    # ...log the test loss
                    writer.add_scalar('loss/test_loss',
                                    test_loss/len(testloader),
                                    global_step=epoch * len(trainloader) + steps)
                    
                    # ...log the test Accuracy
                    writer.add_scalar('test Accuracy',
                                    accuracy/len(testloader),
                                    global_step=epoch * len(trainloader) + steps)

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    writer.add_figure('predictions vs. actuals',
                                    plot_classes_preds(model, inputs, labels),
                                    global_step=epoch * len(trainloader) + steps) 

                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Step :{steps}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(testloader):.3f}.. "
                        f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()

                if steps % args.save_interval==0:
                    path = os.path.join(args.log_dir,"checkpoints",args.name,f"epochs-{epochs}-steps-{steps}")
                    save_checkpoint(path,epoch,model,optimizer,train_losses)
                    print(f"checkpoint saved at :{path}")
                
        
        
        path = os.path.join(args.log_dir,"checkpoints",args.name,"last")
        save_checkpoint(path,epoch,model,optimizer,train_losses)
        print(f"checkpoint saved at :{path}")

    except KeyboardInterrupt:
        path = os.path.join(args.log_dir,"checkpoints",args.name,"last")
        save_checkpoint(path,epoch,model,optimizer,train_losses)
        print(f"Training interrupted checkpoint saved at :{path}")