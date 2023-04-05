import io
from PIL import Image
from torchvision import models
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import urllib
import os

d_r_class = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

def get_model_from_global_agent():
    global_model = models.squeezenet1_1(pretrained=True)
    global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
    global_model.num_classes = 5
    global_model.to(torch.device('cpu'))
    global_model.load_state_dict(torch.load("./model/agg_model.pt", map_location=torch.device('cpu')))
    global_model.eval()
    return global_model


def transform_image(image_bytes):
    apply_transform = transforms.Compose([transforms.Resize(265),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return apply_transform(image).unsqueeze(0)

def get_prediction(input_tensor):
    model = get_model_from_global_agent()
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = int(y_hat.item())
    return d_r_class[predicted_idx]
