from torchvision import models, transforms
import torch
from PIL import Image

def save_model(model, filename):
    # if not os.path.isdir('saved_models'):
    #     os.mkdir('saved_models')
    torch.save(model.state_dict(),  filename+'.pth') 
    print("Model successfully saved.")
    # /content/fasternet_t0-epoch.281-val_acc1.71.9180.pth

def load_model(model, filename):
    model.load_state_dict(torch.load( filename+'.pth'))
    return model

def predict_with_resnet101(image_path):
    
    resnet = models.resnet101()

    resnet = load_model(resnet, 'resnet_pretrained')
    
    # resnet = models.resnet101(pretrained=True)
    
    # resnet = resnet101
    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

'''
def predict_with_resnet18(image_path):
    resnet = models.resnet18 (pretrained=True)

    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
'''