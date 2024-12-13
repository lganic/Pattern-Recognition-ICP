"""
Initial author: AIRocker

Heavily Edited by Logan Boehm
"""

import cv2
from .model import bodypose_model, PoseEstimationWithMobileNet
from .utils.util import*
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import argparse
import os

def create_heatmap(image, backbone = 'Mobilenet'):

    # image = cv2.imread(image)
    # image = cv2.resize(image, (0, 0), fx=1, fy=.3, interpolation=cv2.INTER_CUBIC)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    if backbone == 'CMU':
        model = bodypose_model().to(device)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'weights/bodypose_model'), map_location=lambda storage, loc: storage))
    elif backbone == 'Mobilenet':
        model = PoseEstimationWithMobileNet().to(device)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'weights/MobileNet_bodypose_model'), map_location=lambda storage, loc: storage))
    
    model.eval()

    stride = 8
    padValue = 128
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
    paf_avg = np.zeros((image.shape[0], image.shape[1], 38))
    
    imageToTest = cv2.resize(image, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
    # pad right and down corner to make sure image size is divisible by 8
    im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
    im = np.ascontiguousarray(im)
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)

    with torch.no_grad():
        if backbone == 'CMU':
            Mconv7_stage6_L1, Mconv7_stage6_L2 = model(data)
            _paf = Mconv7_stage6_L1.cpu().numpy()
            _heatmap = Mconv7_stage6_L2.cpu().numpy()
        elif backbone == 'Mobilenet':
            stages_output = model(data)
            _paf = stages_output[-1].cpu().numpy()
            _heatmap = stages_output[-2].cpu().numpy()  
        
    # extract outputs, resize, and remove padding
    heatmap = np.transpose(np.squeeze(_heatmap), (1, 2, 0))  # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    return heatmap

if __name__ == '__main__':

    from matplotlib import pyplot as plt

    heatmap = create_heatmap('../test.png')[:, :, 1]

    plt.imshow(heatmap)
    plt.show()

