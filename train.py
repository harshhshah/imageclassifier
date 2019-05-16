# Import necessary packages 
import torch
from torch import nn
from torch import optim
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


def get_args():
    # Creating Parser
    parser = argparse.ArgumentParser(description="Train Neural Network")
    # Adding arguments
    parser.add_argument('--dir',type=str,action='store',help='Path to Image dataset')
    parser.add_argument('--save_dir',type=str,action='store',help='Path to store checkpoint',default='')
    parser.add_argument('--arch',type=str,action='store',help='Choose model from torchvision.models(densenet121 or vgg16)',default='densenet121')
    parser.add_argument('--learn_rate',type=float,action='store',help='Set learn rate for training',default=0.001)
    parser.add_argument('--hidden_units',type=int,action='store',help='Set size of each hidden layer',default=500)
    parser.add_argument('--epochs',type=int,action='store',help='Set number of epochs for training',default=9)	
    parser.add_argument('--gpu',action='store_true',help='Use GPU for training')

    return parser.parse_args()

# Function for validation
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Function to train
def training(model, train_loader, validation_loader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    steps = 0
    model = model.to(device)
    print("Training Model..")
    for e in range(epochs):
        model.train() #training mode
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval() #evaluation mode
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validation_loader, criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "| Training Loss: {:.4f}".format(running_loss / print_every),
                      "| Validation Loss: {:.3f}.. ".format(validation_loss  / len(validation_loader)),
                      "| Validation Accuracy: {:.3f}%".format(accuracy / len(validation_loader) * 100))
                running_loss = 0
                model.train() #training mode
    print("model trained.")

def main():
    # Get command line arguments
    args = get_args()
    # Extract arguments 
    data_dir = args.dir
    save_dir = args.save_dir
    arch = args.arch
    learn_rate = args.learn_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    print_every = 20
    input_size = 1024 if arch=='densenet121' else 25088
    hidden_size = hidden_units
    output_size = 102

    # Defining transforms for the datasets
    train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    validation_transforms=transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    test_transforms=transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

# Loading the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Defining dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    # Defining model
    model_1 = getattr(models, arch)
    model = model_1(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False

    # Defining  Classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_size)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(p = 0.5)),
                                            ('fc2', nn.Linear(hidden_size, hidden_size //2)),
                                            ('relu2', nn.ReLU()),
                                            ('dropout2', nn.Dropout(p = 0.2)),
                                            ('fc3', nn.Linear(hidden_size //2, output_size)),
                                            ('output', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier
    #criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Training the network
    training(model, train_loader, validation_loader, epochs, print_every, criterion, optimizer, device)
    #checkpoint to save
    checkpoint = {"model":model,
                  "state_dict":model.state_dict(),
                  "optimizer_dict":optimizer.state_dict(),
                  "class_to_idx":train_data.class_to_idx,
                  "classifier":model.classifier,
                  "epochs":epochs,
                  "input_size":input_size,
                  "hidden_size":hidden_size,
                  "output_size":output_size,}
    torch.save(checkpoint, './' + save_dir + '/checkpoint.pth')
    print('Checkpoint is saved')

if __name__ == '__main__':
    main()