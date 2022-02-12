from cv2 import inpaint, reduce
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
import matplotlib.pyplot as plt

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

def dice_loss(input, target):
    target = nn.functional.one_hot(target, num_classes = 10).permute(0,3,1,2).contiguous() 
    smooth = 1.
    
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

epochs = 50       

# # to use cross entropy loss
# criterion = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean") # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

# to use dice coefficient loss
criterion = dice_loss

n_class = 10
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr = 0.015) # choose an optimizer

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # determine which device to use (gpu or cpu)
fcn_model = fcn_model.to(device) #transfer the model to the device
best_model = None

train_loss_epochs = []
val_loss_epochs = []
epoch_list = [i+1 for i in range(epochs)]

class2color = {
    0: (0.4117647058823529, 0.4117647058823529, 0.4117647058823529), \
    1: (0,128,0), \
    2: (0.9137254901960784, 0.5882352941176471, 0.47843137254901963), \
    3: (0,  0,142), \
    4: (0.0, 0.7490196078431373, 1.0),\
    5: (255,255,0), \
    6: (1,0,0), \
    7: (1.0, 0.7137254901960784, 0.7568627450980392),\
    8: (0.8627450980392157, 0.8627450980392157, 0.8627450980392157), \
    9: (0,  0,  0)
}

def train():
    best_iou_score = 0.0
    for epoch in range(epochs):
        ts = time.time()
        losses = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.to(dtype=torch.long)
            
            outputs = fcn_model(inputs) #we will not need to transfer the output, it will be automatically in the same device as the model's!
            
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # backpropagate
            loss.backward()
            # update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        train_loss_epochs.append(np.mean(losses))

        current_miou_score = val(epoch)
        
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            #save the best model
            best_model = torch.save(fcn_model.state_dict(),"best_model.pt")
            
        plt.figure()
        plt.plot(epoch_list, train_loss_epochs, label="Train Loss")
        plt.plot(epoch_list, val_loss_epochs, label="Val Loss")
        plt.xlabel("epochs")
        plt.ylabel("train and val loss")
        plt.legend()
        plt.savefig("plots/train_val_loss.png")
        torch.cuda.empty_cache()

def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device)
            label = label.to(device)
            output = fcn_model(input)
            
            label = label.to(dtype=torch.long)
            
            loss = criterion(output, label)
            losses.append(loss.item()) #call .item() to get the value from a tensor. The tensor can reside in gpu but item() will still work 
            predictions = nn.functional.softmax(output,dim=1)
            pred = predictions.argmax(axis=1)

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))
        
            accuracy.append(pixel_acc(pred, label))


    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    val_loss_epochs.append(np.mean(losses))

    fcn_model.train() #DONT FORGET TO TURN THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)

def test():
    losses = []
    mean_iou_scores = []
    accuracy = []

    best_model = FCN(n_class=n_class)
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load("best_model.pt"))

    #TODO: load the best model and complete the rest of the function for testing
    with torch.no_grad():

        for iter, (input, label) in enumerate(test_loader):

            # both inputs and labels have to reside in the same device as the model's
            input = input.to(device)
            label = label.to(device)
            label = label.to(dtype=torch.long)
            
            output = best_model(input)

            loss = criterion(output, label)
            losses.append(loss.item())
            pred = output.argmax(axis=1)

            mean_iou_scores.append(np.nanmean(iou(pred, label, n_class)))
        
            accuracy.append(pixel_acc(pred, label))
    
    # visualizing the segmented image
    first_image = pred[0]
    segmented_image = np.ones((384,768,3))
    for cls in range(n_class):
        segmented_image[first_image == cls] = class2color[cls]
    plt.imsave("plots/test_segmented.png", segmented_image)

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    test()

    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()