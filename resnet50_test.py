import torch
from torchvision import models
resnet = models.resnet50(pretrained=True)

#gets and opens image *manually switch image for each test
from PIL import Image
img = Image.open("/home/katelynn/Downloads/__MACOSX/img_7_v2.png").convert('RGB')

from torchvision import transforms
# Creates preprocessing pipeline
preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])


# Passes image for preprocessing 
img_preprocessed = preprocess(img)

# Reshapes, crops, and normalizes input tensor 
batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)

resnet.eval()

out = resnet(batch_img_tensor)

# Loads file containing the 1,000 labels 
with open('/home/katelynn/Desktop/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Finds index (tensor) w/the max score in the out tensor

_, index = torch.max(out, 1)

# Finds score percentage w/ torch.nn.functional.softmax function
# normalizes the output to range [0,1] and multiplies by 100
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Prints name w/ score of object identified by model

print(labels[index[0]], percentage[index[0]].item())

# Prints top 5 scores w/image label. Sort function used on torch to sort scores.
# Reads/prints out the categories

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
