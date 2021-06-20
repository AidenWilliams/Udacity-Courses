import argparse
from sys import argv
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

# Create cli arguments
parser = argparse.ArgumentParser()
data_dir = argv[1]
args = argv[2:]
parser.add_argument('--save_dir', action='store',
                    default='checkpoint.pth',
                    dest="save_dir",
                    help='Will store checkpoint after training and testing at save_dir location')

parser.add_argument('--arch', action='store',
                    default='vgg19',
                    dest="arch",
                    help='Architecture that will be loaded from the pytorch model library')

parser.add_argument('--learning_rate', action='store',
                    default=0.003,
                    dest="learning_rate",
                    help='Learning rate for the model to train on')

parser.add_argument('--epochs', action='store',
                    default=5,
                    dest="epochs",
                    help='Number of epochs')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest="gpu",
                    help='Whether to us gpu or not')

# parse remaining args
args = parser.parse_args(args)

# set training and validation directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# define data transforms
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
                                       ])

test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)
                                            ])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


def getModel():
    """
    Function that gets the the specified architecture. Otherwise defaults to vgg19
    :return: vgg19 model with parameters set to false
    """
    pretrained_model = getattr(models, args.arch)
    if callable(pretrained_model):
        model = pretrained_model(pretrained=True)
    else:
        print("Sorry base architecture not recognized, defaulting to vgg19")
        model = models.vgg19(pretrained=True)

    # Freeze parameters so we don't backpropogate through them
    for param in model.parameters():
        param.requires_grad = False

    return model


model = getModel()


class Classifier(nn.Module):
    """
    My custom Classifer deep learning model to be implemented
    """

    def __init__(self):
        """
        Set up the 4 layers, relu and logsoftmax activation functions aswell as dropout rate
        """
        super().__init__()

        # Define hidden layers
        self.fc1 = nn.Linear(25088, 4096, bias=True)
        self.fc2 = nn.Linear(4096, 2048, bias=True)
        self.fc3 = nn.Linear(2048, 1024, bias=True)
        self.fc4 = nn.Linear(1024, 102, bias=True)

        # Define ReLU activation, logsoftmax output and dropout
        self.ReLU = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Define the forward pass of the network
        """
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.logsoftmax(x)

        return x


model.classifier = Classifier()

criterion = nn.NLLLoss()
if args.gpu:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("Cuda not supported on this system, defaulting to CPU usage")
        device = "cpu"
else:
    device = "cpu"

def train(model, epochs=5):
    """
    Function that trains model over the training set, and validates on the validation set
    :param model: model to be trained
    :param epochs: Times training will be done
    :return: trained model
    """
    print("Started training")
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    model.to(device)

    train_losses, valid_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model(inputs)
                    valid_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss / len(train_data))
            valid_losses.append(valid_loss / len(valid_data))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss),
                  "Validation Loss: {:.3f}.. ".format(valid_loss),
                  "Validation Accuracy: {:.3f}".format(accuracy))


train(model, int(args.epochs))


def test(model):
    """
    Tests model over the testing set
    :param model: to be tested
    """
    test_loss = 0
    accuracy = 0

    # Turn off gradients for testing, saves memory and computations
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model(inputs)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Testing Accuracy: {:.3f}".format(accuracy))


test(model)


def SaveModel(model, save_dir):
    torch.save(model.state_dict(), save_dir)


SaveModel(model, args.save_dir)

"""
flowers --save_dir testname.pth --arch vgg11 --learning_rate 0.003 --epochs 6 --gpu
Epoch: 1/6..  Training Loss: 1191.009..  Validation Loss: 121.044..  Validation Accuracy: 0.250
Epoch: 2/6..  Training Loss: 953.864..  Validation Loss: 120.521..  Validation Accuracy: 0.562
Epoch: 3/6..  Training Loss: 985.330..  Validation Loss: 120.099..  Validation Accuracy: 0.562
Epoch: 4/6..  Training Loss: 950.849..  Validation Loss: 119.712..  Validation Accuracy: 0.562
Epoch: 5/6..  Training Loss: 948.361..  Validation Loss: 119.378..  Validation Accuracy: 0.562
Epoch: 6/6..  Training Loss: 945.950..  Validation Loss: 119.079..  Validation Accuracy: 0.562
Testing Accuracy: 0.656
"""