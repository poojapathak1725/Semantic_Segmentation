from cv2 import reduce
from basic_fcn import *
from dataloader import *
from utils import *
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import torch
import gc
from copy import deepcopy
from zipfile import ZipFile

# with ZipFile('tas500v1.1.zip','r') as zip_ref:
#     zip_ref.extractall('')
    
# TODO: Some missing values are represented by '__'. You need to fill these up.
train_dataset = TASDataset('tas500v1.1') 
val_dataset = TASDataset('tas500v1.1', eval=True, mode='val')
test_dataset = TASDataset('tas500v1.1', eval=True, mode='test')


train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= len(test_dataset), shuffle=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases   

epochs = 50       
criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean") # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
n_class = 10
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr = 0.015) # choose an optimizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # determine which device to use (gpu or cpu)
fcn_model = fcn_model.to(device) #transfer the model to the device
best_model = None

def train():
    best_iou_score = 0.0
    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device) #transfer the input to the same device as the model's
            labels = labels.to(device) #transfer the labels to the same device as the model's
            labels = labels.to(dtype=torch.long)
            
            outputs = fcn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            loss = criterion(outputs, labels) #calculate loss
            
            # backpropagate
            loss.backward()
            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        

        current_miou_score = val(epoch)
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            best_model = deepcopy(fcn_model)
            
        torch.cuda.empty_cache()

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.to(device) #transfer the labels to the same device as the model's
            output = fcn_model(input)
            
            label = label.to(dtype=torch.long)
            
            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 
            predictions = nn.functional.softmax(output,dim=1)
            pred = predictions.argmax(axis=1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)

def test():
    losses = []
    mean_iou_scores = []
    accuracy = []

    #TODO: load the best model and complete the rest of the function for testing
    with torch.no_grad():

        for iter, (input, label) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device) #transfer the input to the same device as the model's
            label = label.to(device) #transfer the labels to the same device as the model's
            label = label.to(dtype=torch.long)
            
            output = best_model(input)

            loss = criterion(output, label) #calculate the loss
            losses.append(loss.item())
            pred = output.argmax(axis=1) # Make sure to include an argmax to get the prediction from the outputs of your model

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))  # Complete this function in the util, notice the use of np.nanmean() here
        
            accuracy.append(pixel_acc(pred, label)) # Complete this function in the util


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()