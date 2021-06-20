import argparse
from sys import argv
import numpy as np
import torch
from torchvision import datasets, models
from torch import nn
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()
input = argv[1]
checkpoint = argv[2]
args = argv[3:]

parser.add_argument('--topk', action='store',
                    default=5,
                    dest="topk",
                    help='Number of probabilities to check')

parser.add_argument('--arch', action='store',
                    default='vgg19',
                    dest="arch",
                    help='Architecture that will be loaded from the pytorch model library')

parser.add_argument('--category_names', action='store',
                    default='cat_to_name.json',
                    dest="category_names",
                    help='Category label map')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest="gpu",
                    help='Whether to us gpu or not')

# parse remaining args
args = parser.parse_args(args)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


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


def LoadModel(fullpath):
    """
    Loads checkpoint at fullpath and sets model to the saved model from before
    :param fullpath: full path to checkpoint file
    :return: model
    """
    state_dict = torch.load(fullpath)
    # model must have vgg19 architecture
    pretrained_model = getattr(models, args.arch)
    if callable(pretrained_model):
        model = pretrained_model(pretrained=True)
    else:
        print("Sorry base architecture not recognized, defaulting to vgg19")
        model = models.vgg19(pretrained=True)
    # model must custom classifier as its classifier
    model.classifier = Classifier()
    model.load_state_dict(state_dict)
    return model


model = LoadModel(checkpoint)


def process_image(image):
    # Open the image
    img = Image.open(image)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    return img


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


img = process_image(input)

imshow(img, title="Testing 'Process Image' Function")


def predict(image_path, model, topk=5):
    """
    Get the topk probabilities of what the image at image_path might be, given model
    :param image_path: path to image
    :param model: trained model
    :param topk: number of probabilities to display
    :return: the top k probabilities with their labels and classification
    """
    # Process the image
    img = process_image(image_path)

    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)

    # Make sure the image and model data are both on the same hardware (cpu since no need to use gpu)
    image_tensor.to('cpu')
    model_input.to('cpu')
    model.to('cpu')

    # Run the image through the model
    probs = torch.exp(model.forward(model_input))

    # Get the top probabilities
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]

    train_data = datasets.ImageFolder("flowers/train")

    # Convert indices to classes
    idx_to_class = {val: key for key, val in
                    train_data.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers


def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    # Set up title
    flower_num = image_path.split('/')[2]
    title_ = flower_num
    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title=title_)
    # Make prediction
    probs, labs, flowers = predict(image_path, model, args.topk)
    # Plot bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0])
    plt.show()


plot_solution(input, model)
