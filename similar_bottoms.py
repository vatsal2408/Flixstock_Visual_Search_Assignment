import os
import cv2
import torch
from termcolor import colored
from torchvision import transforms
from utilities.utils import *
from embedding import *
from feature_extractor import *


def load_embed(args, emb_path, txtfile, embed_flag):
    '''
    returns model and image feature embeddings
    '''
    # loads embeddings if it exists
    if os.path.exists(emb_path) and not embed_flag:
        print(colored("Loading Embedding -------", "green"))
        embeddings = torch.load(emb_path)
        model = Model(args).to(args.device)
        return model, embeddings
    else:
        # create embedding if it doesn't exist
        print(colored("Creating Embedding -------", "green"))
        return create_embed(args, txtfile)


def similar_images(args, model, embeddings):
    '''
    takes model and embeddings as input and 
    returns index of desired number of similar images
    '''
    num_images = args.num_images

    # read and preprocess the image for model
    image = cv2.imread(args.input_image, -1)
    image = remove_dummy(image)
    orig = image
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # apply the desired transformation
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
    image = image.unsqueeze(0).to(args.device)

    # creates query embedding
    model.eval()
    with torch.no_grad():
        feat_mat = model(image).cpu()

    feat_mat = feat_mat.reshape((feat_mat.shape[0], -1))
    embeddings = embeddings.reshape((embeddings.shape[0], -1))

    # define the available similarity metrics
    cos = torch.nn.CosineSimilarity(dim=1)
    euc = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

    # compare query embedding to embeddings matrix
    if args.metric == 'euclidian':
        dist = euc(feat_mat, embeddings)
    else:
        dist = cos(feat_mat, embeddings)
    indexes = torch.argsort(dist)[1:num_images+1]
    return orig, indexes