import torch
import sys
import os
import glob
import matplotlib.pyplot as plt
sys.path.append('.')
import clip
from selectivesearch import selective_search
import numpy as np
from skimage import io
import cv2
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root='./checkpoints')



text_label = ["airplane", "ship", "oil tank"]
text_label = [f'It is {desp} in aerial image.' for desp in text_label]
text_label += ['It is an aerial image.']
text_embeds = clip.tokenize(text_label).to(device)


img_path = 'data/Det/multi_label/*.jpg'
img_list = glob.glob(img_path)

for img_name in img_list:
    img = io.imread(img_name)
    h, w, c = img.shape
    img_lbl, regions = selective_search(img, scale=500, sigma=0.9, min_size=100)
    regions = np.random.choice(regions, size=100, replace=False)
    # regions = regions[mask]
    
    roi_imgs = []
    for region in regions:
        l, t, width, height = region['rect']
        if width < 5 or height<5:
            continue
        if width*height > 0.05*h*w:
            continue
        r = l+width
        b = t+height
        roi = img[t:b, l:r, :]
        roi_imgs.append(roi)
    
    img_tensor_list = []
    for roi in roi_imgs:
        img_tensor = preprocess(Image.fromarray(roi.astype('uint8')).convert('RGB')).to(device)
        img_tensor_list.append(img_tensor)
    img_tensor = torch.stack(img_tensor_list)
    with torch.no_grad():
        # image_features = model.encode_image(img_tensor)
        # text_features = model.encode_text(text_embeds)
    
        logits_per_image, logits_per_text = model(img_tensor, text_embeds)
    probs = logits_per_image.softmax(dim=-1)
    pred_labels = probs.argmax(dim=1)
    probs = (probs*100).long()


    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

    print("Label pred:", pred_labels) 


    plt.figure(figsize=(30, 8))
    for idx, img in enumerate(roi_imgs):
        plt.subplot((len(roi_imgs)//6)+1, 6, idx+1)
        img = cv2.resize(img, (128, 128))
        plt.imshow(img)
        prob_per = probs[idx]
        # plt.title(f"{probs[idx]} {text_label[pred_labels[idx]]}", fontsize=10)
        plt.title(f"{probs[idx]}  {pred_labels[idx]}", fontsize=5)
        plt.xticks([])
        plt.yticks([])

    # plt.show()
    plt.savefig(f'results/{os.path.basename(img_name)}', dpi=600)
    # cv2.waitKey(0)