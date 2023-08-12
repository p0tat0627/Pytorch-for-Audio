import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


# download dataset
# create data loader
# build model
# train
# save trained model

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

class FeedForwardNet(nn.Module):
    # constructor of the class
    def __init__(self):
        super().__init__()
        # all the layers of the model 
        # shoudl be initialized here
        # 1 by 1 for the layers
        self.flatten = nn.Flatten() # intial layer layer to flatten the image in 1 dim array
        self.dense_layers = nn.Sequential( # hidden layers
            nn.Linear(28*28, 256), # input layer
            nn.ReLU(), # activation function
            nn.Linear(256, 10) # output layer
        )
        self.softmax = nn.Softmax(dim = 1) # normalization layer finally
    # data flow of the network defined here
    def forward(self, input_data):  # this method tells how to process the data in the model
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data

def create_data_loader(train_data, BATCH_SIZE):
    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        # set input and target to device
        inputs, targets = inputs.to(device), targets.to(device) 

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate error and update
        optimizer.zero_grad() # this is to set gradient to zero after each epoch
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}") # printing the loss of the last batch

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-----------------------------")
    print("Training is done.")


if __name__== "__main__":
    # download MNIST
    train_data, _ = download_mnist_datasets()
    
    # create dataloader
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)

    # build model and assign model to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"using {device} as device")
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)

    # initialize loss function and optimizer for the model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # trian model
    train(feed_forward_net, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    # saving the trianed model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwadnet.pth")




