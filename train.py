'''
train.py - Train a new network on a data set.
           Prints out training loss, validation loss, 
           and validation accuracy as the network trains

usage: train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH] [--learning_rate LR]
                [--hidden_units HIDDEN_UNITS [HIDDEN_UNITS ...]]
                [--epochs EPOCHS] [--dropout DROPOUT] [--gpu]
                [data_dir]
                
positional arguments:
  data_dir

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Path to checkpoint
  --arch ARCH           Pretrained Models : 'vgg16', 'densenet121',
                        'densenet201'
  --learning_rate LR    Network learning rate
  --hidden_units HIDDEN_UNITS [HIDDEN_UNITS ...]
                        3 space separated ints => --hidden_units 512 256 128
  --epochs EPOCHS       Number of epochs or cycles to train the network for
  --dropout DROPOUT     Degree of dropout for regularization
  --gpu                 True if flag --gpu is used
'''
#Import relevant packages
import argparse
import network_utils

#Parse command line arguments using ArgumentParser
parser = argparse.ArgumentParser()

#Store dataset location from command_line
parser.add_argument('data_dir', nargs='?', action="store", default="./flowers")
#Create command line arguments that will be used by train.py:
#--save_dir, --arch, --learning_rate, --hidden_units, --epochs, --gpu, --dropout
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth',\
                    dest="save_dir", action="store",\
                    help='Path to checkpoint')
parser.add_argument('--arch', type=str, default='densenet201',\
                    dest="arch", action="store",\
                    help='Pretrained Models : \'vgg16\', \'densenet121\', \'densenet201\'')
parser.add_argument('--learning_rate', type=float, default=0.003,\
                    dest="lr", action="store",\
                    help='Network learning rate')
parser.add_argument('--hidden_units', type=int, nargs='+', default=[120,100,80],\
                    dest="hidden_units", action="store",\
                    help='3 space separated ints => --hidden_units 512 256 128')
parser.add_argument('--epochs', type=int, default=1,\
                    dest="epochs", action="store",\
                    help='Number of epochs or cycles to train the network for')
parser.add_argument('--dropout', type=float, default=0.2,\
                    dest="dropout", action="store",\
                    help='Degree of dropout for regularization')
parser.add_argument('--gpu', dest="gpu", action="store_true",\
                    default=False, help='True if flag --gpu is used')

collection = parser.parse_args()

data_dir = collection.data_dir
cp_dir = collection.save_dir
pt_network = collection.arch
lr = collection.lr 
hidden_count = collection.hidden_units
epochs = collection.epochs
p = collection.dropout
is_gpu = collection.gpu

#print(collection)

#Get train, validate and test data loader generators
#Also get train_dataset for checkpoint creation later
trainloader, validateloader, testloader, train_dataset = network_utils.get_dataloaders(data_dir)
#Get Pre-trained network model, criterion and optimizer
model, criterion, optimizer  = network_utils.set_network_params(pt_network, hidden_count, p, lr, is_gpu)
#Train and validate newly created network with test and validation data respectively
network_utils.train_and_validate_network(model, criterion, optimizer, trainloader, validateloader, epochs, is_gpu)
#Test the accuracy of trained model
network_utils.network_test_accuracy_checker(model, criterion, optimizer, testloader, is_gpu)
#Create and save checkpoint
network_utils.save_checkpoint(model, train_dataset, pt_network, hidden_count, cp_dir)
