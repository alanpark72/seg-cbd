from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from natsort import natsorted
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from utils import metrics, dataloader
from utils.utils import checkDir, getConfig, mask2rgb
from model.res_unet import ResUnet
from model.res_unet_plus import ResUnetPlusPlus
import torch
import argparse
import os

def loadBatch(path):
    _test_imgs = natsorted(glob("{}images/*.jpg".format(path)))
    _test_lbls = natsorted(glob("{}labels/*.png".format(path)))
    #_test_imgs = natsorted(glob("{}*/*/*.jpg".format(path))) # for origin dataset
    #_test_lbls = natsorted(glob("{}*/*/mask/*.png".format(path))) # for origin dataset

    print("Test Images : {}(ea)".format(len(_test_imgs)))
    print("Test Labels : {}(ea)".format(len(_test_lbls)))

    #_transform = transforms.Compose([Normalization(0.5,0.5), Resize(config["imgsz"]), ToTensor()])
    _transform = transforms.Compose([dataloader.ToTensorTarget()])
    _test_data = dataloader.Dataset(_test_imgs, _test_lbls, transform=_transform)
    #_test_data = Dataset(_test_imgs, _test_lbls, transform=_transform)
    test_batches = DataLoader(_test_data, batch_size=1, shuffle=False, num_workers=24, pin_memory=True)
    
    return test_batches

def loadModel(weight, use_cuda, is_pp=False):
    if is_pp:
        model = ResUnetPlusPlus(3).cuda()
    else:
        model = ResUnet(3, 64).cuda()
    
    model.load_state_dict(torch.load(weight)["state_dict"])
    model.eval()
    
    if use_cuda:
        model = model.cuda()
        model.eval()

    #weight_name = weight.split("/")[-1]
    #num = weight_name.split("_")[-1][:-8]
    
    return model

def overlayMask(input, mask, is_pred=False):
    if is_pred:
        #mask = filterPrediction(mask)
        rst = mask2rgb(mask, (512,512), True)
        rst = cv2.addWeighted(input, 0.5, rst, 1.0, 1.0)
        return rst
    else:
        rst = cv2.addWeighted(input, 0.5, mask, 1.0, 1.0)
        return rst

def postProcess(preds, test_batches, save_path):
    ls_dice = []

    for batch, data in enumerate(tqdm(test_batches, desc="post-proc")):
        pred = preds[batch]
        _input_name = data["name"][0]
        _input = np.asarray(Image.open(_input_name).convert('RGB'))
        _label_name = _input_name.replace("images", "labels").replace(".jpg", ".png")
        _label = np.asarray(Image.open(_label_name).convert('RGB'))
        
        
        pred_path = "{}pred/".format(save_path)
        checkDir(pred_path)
        label_path = "{}label/".format(save_path)
        checkDir(label_path)
        input_path = "{}input/".format(save_path)
        checkDir(input_path)
        eval_path = "{}eval/".format(save_path)
        checkDir(eval_path)
        crop_path = "{}crop/".format(save_path)
        checkDir(crop_path)
        
        _ov_label = overlayMask(_input, _label)
        
        for i in range(test_batches.batch_size):
            #pred_img = mask2rgb(pred[i].cpu(), (512,512))
            _name,_ext = os.path.splitext(os.path.basename(data["name"][i]))
            _ov_pred = overlayMask(_input, pred[i].cpu(), True)
            _ov_eval = overlayMask(_ov_label, _ov_pred)
            #_crop = cropCBD(_input, pred[i].cpu())
            plt.imsave(pred_path+"{}.png".format(_name), _ov_pred)
            ##plt.imsave(label_path+"{}.png".format(_name), _ov_label)
            ##plt.imsave(input_path+"{}.png".format(_name), _input)
            plt.imsave(eval_path+"{}.png".format(_name), _ov_eval)
            #cv2.imwrite(crop_path+"{}.png".format(_name), _crop)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()

config = getConfig("./configs/cbd.yaml")

model = loadModel(config["weight"],use_cuda,config["RESNET_PLUS_PLUS"])

result_path = "./result/"
checkDir(result_path)

# set up binary cross entropy and dice loss
criterion = metrics.BCEDiceLoss()

# optimizer
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# decay LR
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# starting params
best_loss = 999
start_epoch = 0

# get data
_test_batches = loadBatch(config["test"])
lst_pred = []

# creating loaders
with torch.no_grad():
    for num, data in enumerate(tqdm(_test_batches, desc="test")):
        print("Start Segmenting for #{} batch".format(num))
        print("-" * 60)
        test_acc = metrics.MetricTracker()
        
        input = data["input"].to(device)
        target = data["label"].to(device)
        
        output = model(input)
        
        #prob = torch.sigmoid(output)
        #pred = torch.argmax(prob, dim=1)
        pred = torch.argmax(output, dim=1)
        
        lst_pred.append(pred)
        
        test_acc.update(metrics.dice_coeff(output, target), output.size(0))
        print(test_acc.avg)
        
postProcess(lst_pred, _test_batches, result_path)