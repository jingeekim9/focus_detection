import numpy as np
import pandas as pd
import base64
import cv2
import os
import io
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms
import torch.nn as nn
import torch
from PIL import Image
import matplotlib.pyplot as plt

image = cv2.imread('sleepy_driver.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.resize(image, (224, 244))
plt.imshow(image)
plt.show()
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load('model/focus_model.pth', map_location=torch.device('cpu')))
X = image.astype(np.float32)
X = X/255
transform = transforms.ToTensor()
X = transform(X)
X = X.unsqueeze(0)
model.eval()
pred = model(X)
output = torch.topk(pred[0], 1)[1][0].item()
if output == 1:
  print("Unfocused")
else:
  print("Focused")
