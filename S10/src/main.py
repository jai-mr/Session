import torch
import torch.optim as optim
import matplotlib.pyplot as plt
cuda = torch.cuda.is_available()
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR

from models import *
from utilities import *
from train_loss import *
from test_loss import *

def main_s10(num_epochs, lr):

    # Data Augmentation & data loader stuff to be handled
    batch_size = 128
    trainloader, testloader = S10_CIFAR10_data_prep(batch_size)

    # Creating tensorboard writer
    img_save_path = '/content/gdrive/MyDrive/EVA8/s10_Final/logs/'
    
    #print("Image Path: " , img_save_path)
    tb_writer = create_tensorboard_writer(img_save_path)
    #print("Tensor Board: " , tb_writer)
    
    # Creating plot object
    plot = cifar10_plots(img_save_path, tb_writer)

    # Displaying train data
    plot.plot_cifar10_train_imgs(trainloader)

    # Displaying torch summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = ViT(image_size=32,patch_size= 4,num_classes=10, dim=49,depth=6,heads=8,mlp_dim=147,numb_patch=8, 
                dropout=0.1,emb_dropout=0.1)
    model = model.to(device)
    summary(model, input_size=(3, 32, 32))

    # Adding model graph to tensor-board
    img = torch.ones(1, 3, 32, 32)
    img = img.to(device)
    tb_writer.add_graph(model, img)

    # Training the model for fixed epochs
    EPOCHS = num_epochs
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    stats = ctr()
    train = train_losses(model, device, trainloader, stats, optimizer, EPOCHS)
    test  = test_losses(model, device, testloader, stats, EPOCHS)

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch+1}')
        train.s10_train(epoch, tb_writer)
        test.s10_test(epoch, tb_writer)

    details = counters

    # Displaying 20 misclassified images
    num_images = 25
    plot.plot_cifar10_misclassified(details, num_images)

    # Plotting train & test accuracies and losses
    plt.figure(figsize=(12, 8))
    plt.title(f"Train Losses")
    plt.plot(details['train_loss'])

    
    plt.savefig(f'{img_save_path}train_loss.jpg')
    
    
    plt.figure(figsize=(12,8))
    plt.title(f"Train Accuracy")
    plt.plot(details['train_acc'])
    plt.savefig(f'{img_save_path}train_acc.jpg')
    
    plt.figure(figsize=(12,8))
    plt.title(f"Test Losses")
    plt.plot(details['test_loss'])
    plt.savefig(f'{img_save_path}test_loss.jpg')
    
    plt.figure(figsize=(12,8))
    plt.title(f"Test Accuracy")
    plt.plot(details['test_acc'])
    plt.savefig(f'{img_save_path}test_acc.jpg')

    return f' main_s10() ended successfully '