from torchvision import models, transforms
import torch
from PIL import Image
import numpy as np
import random
import cv2
import os
import gdown

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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
    
    # resnet = models.resnet101()

    # resnet = load_model(resnet, 'resnet_pretrained')
    

    if os.path.isfile('resnet_scripted.pt')==False:


        URL_RESNET = "https://drive.google.com/uc?id=18Rh9mN8FXb-8gsq9KsC9pxE3boqLrpFH"
        
        output = 'resnet_scripted.pt'
        gdown.download(URL_RESNET, output, quiet=False)

        # os.system("gdown 18Rh9mN8FXb-8gsq9KsC9pxE3boqLrpFH")
        
        # response_res = requests.get(URL_RESNET)
        # open("resnet_scripted.pt", "wb").write(response_res.content)

    resnet = torch.jit.load('resnet_scripted.pt')
    print(type(resnet))
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

def random_color_masks(image):
  # I will copy a list of colors here
  colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, 10)]
  colored_mask = np.stack([r,g,b], axis=2)
  return colored_mask

def segment_with_mrcnn(image_path):
    

    if os.path.isfile('mrcnn_scripted.pt')==False:


    #     # URL_MRCNN = "https://drive.google.com/file/d/1E86dc0S3gx8P0Prtm1yqNVG5OuDnOtqX/view?usp=sharing"
    # https://drive.google.com/file/d/1E86dc0S3gx8P0Prtm1yqNVG5OuDnOtqX/view?usp=sharing
        URL_MRCNN = "https://drive.google.com/uc?id=1E86dc0S3gx8P0Prtm1yqNVG5OuDnOtqX"
        
        # os.system("gdown 1E86dc0S3gx8P0Prtm1yqNVG5OuDnOtqX")
        output = 'mrcnn_scripted.pt'
        gdown.download(URL_MRCNN, output, quiet=False)

        # response_mrcnn = requests.get(URL_MRCNN)
        # open("mrcnn_scripted.pt", "wb").write(response_mrcnn.content)

    mrcnn = torch.jit.load('mrcnn_scripted.pt')
    print(type(mrcnn))
    
    # mrcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    threshold=0.5
    
    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    # batch_t = torch.unsqueeze(transform(img), 0)

    mrcnn.eval()

    _, pred = mrcnn([transform(img)]) # Send the image to the model. This runs on CPU, so its going to take time
    #Let's change it to GPU
    # pred = pred.cpu() # We will just send predictions back to CPU
    # Now we need to extract the bounding boxes and masks
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    preds = [pred_score.index(x) for x in pred_score if x > threshold]
    print(len(preds))
    
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]

    # return masks, pred_boxes, pred_class

    rect_th=3
    text_size=3
    text_th=3
    
    print(type(img))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For working with RGB images instead of BGR
    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i])
        print("--", type(img), img.shape)
        print("--", type(rgb_mask), rgb_mask.shape)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        pt1 = tuple(int(x) for x in pred_boxes[i][0])
        pt2 = tuple(int(x) for x in pred_boxes[i][1])
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_class[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    return img, pred_class, masks[i]

    

    # out = resnet(batch_t)

    # with open('imagenet_classes.txt') as f:
    #     classes = [line.strip() for line in f.readlines()]

    # prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # _, indices = torch.sort(out, descending=True)
    # return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
