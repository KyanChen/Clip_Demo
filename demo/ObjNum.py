import torch
import sys
import os
import glob
import matplotlib.pyplot as plt
sys.path.append('.')
import clip
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root='./checkpoints')

img_path = 'data/Det/single_label/*.jpg'
img_list = glob.glob(img_path)

img_tensor_list = []
for img in img_list:
    img_tensor_list.append(preprocess(Image.open(img).convert("RGB")).to(device))
img_tensor = torch.stack(img_tensor_list)

text_label = list(range(0, 15))
text_desps = [f'There are {repr(desp)} planes parked at the airport.' for desp in text_label]
text_desps[0] = 'There is no plane.'
text_desps[1] = 'There is a plane at the airport.'
text_embeds = clip.tokenize(text_desps).to(device)



with torch.no_grad():
    image_features = model.encode_image(img_tensor)
    text_features = model.encode_text(text_embeds)
    
    logits_per_image, logits_per_text = model(img_tensor, text_embeds)
    probs = logits_per_image.softmax(dim=-1)
    pred_labels = probs.argmax(dim=1)
    probs = (probs*100).long()


print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

print("Label pred:", pred_labels) 


plt.figure(figsize=(20, 4))
for idx, img in enumerate(img_list):
    image = Image.open(img).convert("RGB")

    plt.subplot(3, 5, idx+1)
    plt.imshow(image)
    prob_per = probs[idx]
    # plt.title(f"{probs[idx]} {repr(text_label[pred_labels[idx]])}", fontsize=10)
    plt.title(f"NUM: {repr(text_label[pred_labels[idx]])}", fontsize=10)
    plt.xticks([])
    plt.yticks([])

plt.show()