import os
import torch
from torchvision import transforms
from pretrained_model import Model
from Dataset import MyDataset


def create_embed(args, txtfile):
    '''
    Creates the feature embeddings for image dataset
    '''

    # preprocess the data for model
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), 
                                                            (0.229, 0.224, 0.225))
                                ])
    dataset = MyDataset(args, txtfile, transform)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size)

    # load the model
    model = Model(args).to(args.device)
    model.eval()

    demo_tnsr = torch.randn((1, 3, 224, 224)).to(args.device)
    demo = model(demo_tnsr)
    embeddings = torch.randn_like(demo).cpu()
    
    # iterate over images to create embeddings
    for images in dataloader:
        images = images.to(args.device)
        
        feat_mat = model(images).cpu()
        
        embeddings = torch.cat((embeddings, feat_mat), 0)
    
    # saves the embeddings
    torch.save(embeddings, os.path.join(args.root_dir, 'embeddings\{0}_embeddings.pt'.format(args.model)))

    return model, embeddings