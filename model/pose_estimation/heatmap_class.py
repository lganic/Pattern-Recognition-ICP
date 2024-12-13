import cv2
from .model import bodypose_model, PoseEstimationWithMobileNet
from .utils.util import *
import torch
import numpy as np
import os


class HeatmapGenerator:
    def __init__(self, backbone='Mobilenet'):
        """
        Initialize the HeatmapGenerator class with the specified backbone model.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone

        if backbone == 'CMU':
            self.model = bodypose_model().to(self.device)
            weight_path = os.path.join(os.path.dirname(__file__), 'weights/bodypose_model')
        elif backbone == 'Mobilenet':
            self.model = PoseEstimationWithMobileNet().to(self.device)
            weight_path = os.path.join(os.path.dirname(__file__), 'weights/MobileNet_bodypose_model')
        else:
            raise ValueError("Invalid backbone. Choose 'CMU' or 'Mobilenet'.")

        self.model.load_state_dict(torch.load(weight_path, map_location=lambda storage, loc: storage))
        self.model.eval()

    def create_heatmap(self, image):
        """
        Generate a heatmap for the given input image using the loaded model.
        """
        stride = 8
        padValue = 128
        heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
        paf_avg = np.zeros((image.shape[0], image.shape[1], 38))

        imageToTest = cv2.resize(image, (0, 0), fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
        im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.backbone == 'CMU':
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(data)
                _paf = Mconv7_stage6_L1.cpu().numpy()
                _heatmap = Mconv7_stage6_L2.cpu().numpy()
            elif self.backbone == 'Mobilenet':
                stages_output = self.model(data)
                _paf = stages_output[-1].cpu().numpy()
                _heatmap = stages_output[-2].cpu().numpy()

        heatmap = np.transpose(np.squeeze(_heatmap), (1, 2, 0))
        heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        return heatmap


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Example usage
    generator = HeatmapGenerator(backbone='Mobilenet')
    input_image = cv2.imread('../test.png')  # Replace with your test image path
    heatmap = generator.create_heatmap(input_image)[:, :, 1]

    plt.imshow(heatmap)
    plt.show()
