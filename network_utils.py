'''
network_utils.py - Performs loading of datasets,
                   sets network parameters,
                   Trains the model and validates the training,
                   tests the accuracy of trained model,
                   saves checkpoint,
                   loads checkpoint,
                   processes an image,
                   predicts most probable image categories for a given image
'''
#Import relevant packages
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import keep_awake, active_session
from PIL import Image
import numpy as np

#Global settings

#Define a dictionary mapping of Pretrained Networks to a list of
#their input features and torchvision command to get the respective models.
nn_list = {'vgg16' : [25088, "models.vgg16(pretrained=True)"],
           'densenet121' : [1024, "models.densenet121(pretrained=True)"],
           'densenet201' : [1920, "models.densenet201(pretrained=True)"]}

#Number of flower classes form the output layer features
output_count = 102

#Set device that is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(data_dir):
    '''
    Perform training data augmentation, data normalization, loading and batching
    for the training, validation, and testing sets
    '''
    #Define dataset directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    validate_transform = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    validate_dataset = datasets.ImageFolder(valid_dir, transform=validate_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validate_dataset, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    return trainloader, validateloader, testloader, train_dataset

def set_network_params(pt_network, hidden_count, p, lr, is_gpu):
    '''
    set_network_params - sets the network parameters from command line or default from train.py's argument parser
                           pt_network - sets a pretrained neural network model provided by user.
                           hidden_count - sets hidden layer features provided by user in custom classifier
                           p - sets dropout rate provided by user in custom classifier
                           lr - sets learning rate provided by user
                           is_gpu - True if processor is gpu
                       - defines a custom feedforward classifier with 3 hidden layers
                       - selects available device - cpu or cuda
                       - sets loss criterion
                       - defines an optimizer using learning rate provided by user.
    returns model, criterion and optimizer definitions to be used by training module
    '''
    input_count = nn_list[pt_network][0]
    model = eval(nn_list[pt_network][1])

    #Turn of autograd
    for param in model.parameters():
        param.requires_grad=False

    #Define custom feedforward classifier "Netifier"
    Netifier = nn.Sequential(OrderedDict([('dropout', nn.Dropout(p)),
                                          ('input', nn.Linear(input_count, hidden_count[0])),
                                          ('relu1', nn.ReLU()),
                                          ('hidden1', nn.Linear(hidden_count[0], hidden_count[1])),
                                          ('relu2', nn.ReLU()),
                                          ('hidden2', nn.Linear(hidden_count[1], hidden_count[2])),
                                          ('relu3', nn.ReLU()),
                                          ('hidden3', nn.Linear(hidden_count[2], output_count)),
                                          ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = Netifier

    #Move the newely defined model to available device
    if str(device)=='cuda' and is_gpu:
        model.to(device);
    #else:#cpu
        #model.to(device);

    #Define criterion
    criterion = nn.NLLLoss()

    #Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer

def validate_network(model, criterion, optimizer, dataloader, is_gpu):
    '''
    validate_network - Runs validation loop for either validation or test dataset based on dataloader
                       Indicates how well our model is training.
                       Inputs:
                       model, criterion, optimizer, dataloader, is_gpu
                       Returns Loss and Accuracy
    '''
    loss = 0
    accuracy = 0
    #Turn off gradient to speed up process
    with torch.no_grad():
        #Turn off dropout
        model.eval()
        for images, labels in dataloader:
            #Clear the accumulated gradient
            optimizer.zero_grad()
            #Load data to available device
            if str(device)=='cuda' and is_gpu:
                images, labels = images.to(device), labels.to(device)
            #Do a forward pass and get log probability
            output = model.forward(images)
            #Compute running loss
            loss += criterion(output, labels).item()
            #Get linear probability
            prob_output = torch.exp(output)
            #Get highest probability and its class
            top_p, top_class = prob_output.topk(1, dim=1)
            #Compare top class with actual label
            equals = top_class == labels.view(*top_class.shape)
            #Take the average of accumulated accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        #Turn dropout on
        model.train()
    return loss, accuracy

def train_and_validate_network(model, criterion, optimizer, trainloader, validateloader, epochs, is_gpu):
    '''
    train_and_validate_network - Runs Training loop with training dataset and computes training loss.
                                 Every fixed interval performs validation to compute testing loss and accuracy
                                 Inputs:
                                 model, criterion, optimizer, trainloader, validateloader, epochs, is_gpu
                                 Output:
                                 Prints Training loss, validation loss and validation accuracy
    '''
    with active_session():
        print("\n***Model Training Started!***\n")
        step=0
        print_interval=10
        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:
                step += 1
                #Load data to available device
                if str(device)=='cuda' and is_gpu:
                    images, labels = images.to(device), labels.to(device)
                #Clear the accumulated gradient
                optimizer.zero_grad()
                #Do a forward pass and get log probability
                output = model.forward(images)
                #Compute loss
                loss = criterion(output, labels)
                #Do a backward pass and get the gradients
                loss.backward()
                #Update weights and biases
                optimizer.step()
                #Compute running loss
                running_loss += loss.item()
                if step%print_interval==0:
                    #Validate network performance and get loss and accuracy
                    vloss, vaccuracy = validate_network(model, criterion, optimizer, validateloader, is_gpu)
                    print(f"Epoch {e+1}/{epochs} : "
                          f"Training Loss = {running_loss/len(trainloader):.3f}, "
                          f"Validation Loss = {vloss/len(validateloader):.3f}, "
                          f"Validation Accuracy = {vaccuracy/len(validateloader)*100:.3f}%")

        print("\n***Hurray! Model Training complete***\n")


def network_test_accuracy_checker(model, criterion, optimizer, testloader, is_gpu):
    '''
    Perform validation on the test set and print model's accuracy
    '''
    _,test_accuracy = validate_network(model, criterion, optimizer, testloader, is_gpu)
    accuracy = test_accuracy/len(testloader)*100
    print(f"Trained Network Model's Accuracy : {accuracy:.3f}%")
    if (accuracy>=70):
        print("Model is well trained!")
    else:
        print("Retrain model!")


def save_checkpoint(model, train_dataset, pt_network, hidden_count, cp_dir):
    '''
    Save the checkpoint
    '''
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'model_name' : pt_network,
                  'hidden_layers_size': hidden_count,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, cp_dir)

def load_checkpoint(cp_file, is_gpu):
    '''
    Load checkpoint and rebuild the model
    '''
    checkpoint = torch.load(cp_file, map_location=device)
    model_name = checkpoint['model_name']
    model,_,_ = set_network_params(model_name, checkpoint['hidden_layers_size'], 0.2, 0.03, is_gpu)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    Input - Full image path
    Returns a torch tensor
    '''
    pil_image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    final_image = transform(pil_image)
    return final_image

def predict(image_path, model, top_k, cat_to_name, is_gpu):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    Input image can be from the downloaded dataset 'flowers'
    or any image that is not part of the dataset, but saved in ideal size and format.
    '''
    #Computation does not require GPU
    #device = "cpu" #for testing purpose
    model.to(device)
    model.eval()
    #Preprocess image
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()
    #Make a forward pass with the image and get top-K probabilities and flower labels
    with torch.no_grad():
        if (str(device)=="cpu"):
            output = model.forward(img_torch)
        elif (is_gpu or str(device)=='cuda'):
            output = model.forward(img_torch.cuda())
    output = torch.exp(output)
    probs, labels = output.topk(top_k)

    #Get top flowers from top labels through class_to_idx
    top_probs = np.array(probs.detach())[0]
    top_labels = np.array(labels.detach())[0]
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_labels = [idx_to_class[l] for l in top_labels]
    top_flowers = [cat_to_name[l] for l in top_labels]

    #Only if the sample image is from local flower dataset,
    #get the actual name of the flower
    print_verdict = False
    #It would be ideal to have data_dir from train script here
    #instead of using the name 'flowers'
    if(image_path.split('/')[0] == "flowers"):
        flower_num = image_path.split('/')[2]
        title = cat_to_name[flower_num]
        title.replace('_',' ')
        print_verdict=True

    print('\n\n---------------------------------------------')
    print(f'PREDICTION RESULT \nImage : {image_path}')
    print(f'DISPLAY TOP {top_k} MOST PROBABLE FLOWER CATEGORIES\nAND THEIR PROBABILITIES')
    print('---------------------------------------------')
    print('FLOWER CATEGORY\t\tPROBABILITY')
    print('---------------------------------------------')
    for i, j in zip(top_flowers, top_probs):
        print(f'{i:20} :\t {j:.3f}')
    print('---------------------------------------------')
    if(print_verdict):
        print('VERDICT:')
        if title == top_flowers[0]:
            print(f'Actual flower name {title.title()} matches the \ntop prediction by our model!')
        else:
            print(f'Actual flower name {title.title()} does not match \nthe top prediction by our model!\
                    \nRetrain model for better prediction!')
        print('---------------------------------------------')
