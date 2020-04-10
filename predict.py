'''
predict.py - Predict flower name from an image along with the probability of that name.

usage: predict.py [-h] [--input IMAGE_FILE] [--checkpoint CP_FILE]
                  [--top_k TOP_K] [--category_names CAT2NAME_FILE] [--gpu]

optional arguments:
  -h, --help            show this help message and exit
  --input IMAGE_FILE    Path to sample image file
  --checkpoint CP_FILE  Checkpoint file path
  --top_k TOP_K         Top k most likely flower classes
  --category_names CAT2NAME_FILE
                        File mapping categories to real flower names
  --gpu                 True if flag --gpu is used
'''
#Import relevant packages
import json
import argparse
import network_utils

#Parse command line arguments using ArgumentParser
parser = argparse.ArgumentParser()

#Create command line arguments that will be used by predict.py:
#--input, --checkpoint, --top_k, --category_names, --gpu
parser.add_argument('--input', type=str, default='flowers/test/90/image_04469.jpg',\
                    dest="image_file", action="store",\
                    help='Path to sample image file')
parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth',\
                    dest="cp_file", action="store",\
                    help='Checkpoint file path')
parser.add_argument('--top_k', type=int, default=5,\
                    dest="top_k", action="store",\
                    help='Top k most likely flower classes')
parser.add_argument('--category_names', type=str, default='./cat_to_name.json',\
                    dest="cat2name_file", action="store",\
                    help='File mapping categories to real flower names')
parser.add_argument('--gpu', dest="gpu", action="store_true",\
                    default=False, help='True if flag --gpu is used')

collection = parser.parse_args()

image_file = collection.image_file
cp_file = collection.cp_file
top_k = collection.top_k
cat2name_file = collection.cat2name_file 
is_gpu = collection.gpu

#print(collection)

#Load saved checkpoint
model = network_utils.load_checkpoint(cp_file, is_gpu)

#Load the dictionary mapping flower label integer to actual names of flowers
with open(cat2name_file, 'r') as f:
    cat_to_name = json.load(f)

#Predict top_k most probable categories for a sample flower
network_utils.predict(image_file, model, top_k, cat_to_name, is_gpu)
