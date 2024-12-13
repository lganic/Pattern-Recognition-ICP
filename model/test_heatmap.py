from matplotlib import pyplot as plt
import cv2
import numpy as np
from pose_estimation.create_heatmap import create_heatmap
from PIL import Image

def overlay_heatmap_on_image(image, heatmap, colormap=cv2.COLORMAP_JET, alpha=.5):
    """
    Overlay a single-channel heatmap on an image using a colormap and alpha blending.
    
    Parameters:
    - image: Original image as a NumPy array (BGR).
    - heatmap: Single-channel heatmap as a NumPy array.
    - colormap: OpenCV colormap to apply.
    - alpha: Transparency factor for the heatmap.
    
    Returns:
    - Composite image with heatmap overlay.
    """

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert image to BGR if it's in RGB
    if image.shape[2] == 3:
        image_bgr = image.copy()
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Blend the heatmap with the image
    composite = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return composite

image_name = 'simple.jpg'

plt.subplot(1, 2, 1)

image = cv2.imread(image_name)

im_for_size = Image.open(image_name)
plt.imshow(im_for_size)

heatmap = 1 - create_heatmap(image)[:, :, 1]

# heatmap = cv2.resize(heatmap, im_for_size.size)

print(type(heatmap))

plt.title('Original Image')
plt.axis('off')

overlay = overlay_heatmap_on_image(image, heatmap)


plt.subplot(1, 2, 2)

plt.title('Heatmap Overlay (Chest)')
plt.axis('off')

plt.imshow(overlay)

plt.show()

plt.subplot(1,3,1)
plt.title('Left Hand Heatmask')


heatmaps = create_heatmap(image)

heatmap = 1 - heatmaps[:, :, 4]

plt.imshow(overlay_heatmap_on_image(image, heatmap))

plt.subplot(1,3,2)
plt.title('Right Hand Heatmask')


heatmaps = create_heatmap(image)

heatmap = 1 - heatmaps[:, :, 7]

plt.imshow(overlay_heatmap_on_image(image, heatmap))

plt.subplot(1,3,3)
plt.title('Combined Hand Heatmask')


heatmaps = create_heatmap(image)

heatmap = (1 - heatmaps[:, :, 4]) + (1 - heatmaps[:, :, 7])

plt.imshow(overlay_heatmap_on_image(image, heatmap))

plt.show()