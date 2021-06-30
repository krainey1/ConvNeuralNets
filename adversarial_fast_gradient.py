import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable

#gets model
inceptionv3 = models.inception_v3(pretrained=True)
img = Image.open("/home/katelynn/Downloads/__MACOSX/img_4_v1.png").convert('RGB')
inceptionv3.eval()

preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img_tensor = preprocess(img) 
img_tensor = img_tensor.unsqueeze(0)

img_var = Variable(img_tensor, requires_grad=True)

output = inceptionv3.forward(img_var)
label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element
print(label_idx)
   
with open('/home/katelynn/Desktop/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
img_out = labels[label_idx]
print(img_out)

#calculates probability
_, index = torch.max(output, 1)
prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
percentage = round(prob[index[0]].item(), 4)
print(percentage)

#Fast gradient sign method

y_true = label_idx 
target = Variable(torch.LongTensor([y_true]), requires_grad=False)

loss = torch.nn.CrossEntropyLoss()
loss_cal = loss(output, target)
loss_cal.backward(retain_graph=True) 

#higher epsilon value = more preturbation in the image = change in class *smaller values change percentage but not class
eps = 0.90
grad = torch.sign(img_var.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
adversarial = img_var.data + eps * grad          #find adv example using formula shown above
output_adv = inceptionv3.forward(Variable(adversarial))   #perform a forward pass on adv example
x_adv = labels[torch.max(output_adv.data, 1)[1][0]]    #classify the adv example
#op_adv_prob = F.softmax(output_adv, dim=1)                 #get probability distribution over classes
#adv_prob =  (torch.max(op_adv_prob.data, 1)[0][0]) * 100      #find probability (confidence) of a predicted class
_, index1 = torch.max(output_adv, 1)
op_adv_prob = torch.nn.functional.softmax(output_adv, dim=1)[0] * 100
adv_percentage = round(op_adv_prob[index1[0]].item(), 4)


print(x_adv)
print(adv_percentage)

