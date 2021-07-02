import torch
import torch.nn
import torch.autograd.gradcheck
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import torch.optim

#Iterative Target Class method
def iterative_target(label_idx, labels, output, img_var, image_tensor, resnet):
    y_true = label_idx
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)

    eps = 1.0
    
    num_steps = 5
    alpha = 0.025

    for i in range(num_steps):
        resnet.zero_grad()
        output  = resnet.forward(img_var)
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, target)
        loss_cal.backward()
        grad = alpha * torch.sign(img_var.grad.data)   
        adv_temp = img_var.data + grad                 #add perturbation to img_variable which also contains perturbation from previous iterations
        total_grad = adv_temp - image_tensor                  #total perturbation
        total_grad = torch.clamp(total_grad, -eps, eps)
        adv = image_tensor + total_grad                      #add total perturbation to the original image
        img_var.data = image_tensor

    output_adv = resnet.forward(Variable(adv))  
    x_adv = labels[torch.max(output_adv.data, 1)[1][0]]    
    _, index1 = torch.max(output_adv, 1)
    op_adv_prob = torch.nn.functional.softmax(output_adv, dim=1)[0] * 100
    adv_percentage = round(op_adv_prob[index1[0]].item(), 4)

    return [x_adv, adv_percentage]

#Basic Iterative Method
def basic_iterative(label_idx, labels, output, img_var, image_tensor, resnet):
    y_true = label_idx
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)
    
    eps = 0.90
    num_steps = 5
    alpha = 0.025

    for i in range(num_steps):
        resnet.zero_grad()
        output  = resnet.forward(img_var)
        loss = torch.nn.CrossEntropyLoss()
        loss_cal = loss(output, target)
        loss_cal.backward()
        grad = alpha * torch.sign(img_var.grad.data)   
        adv_temp = img_var.data + grad                 #add perturbation to img_variable which also contains perturbation from previous iterations
        total_grad = adv_temp - image_tensor                  #total perturbation
        total_grad = torch.clamp(total_grad, -eps, eps)
        adv = image_tensor + total_grad                      #add total perturbation to the original image
        img_var.data = adv

    output_adv = resnet.forward(Variable(adv))  
    x_adv = labels[torch.max(output_adv.data, 1)[1][0]]    
    _, index1 = torch.max(output_adv, 1)
    op_adv_prob = torch.nn.functional.softmax(output_adv, dim=1)[0] * 100
    adv_percentage = round(op_adv_prob[index1[0]].item(), 4)

    return [x_adv, adv_percentage]

        
                      
    
    
#One-step target class method

def one_step_target(label_idx, labels, output, img_var, resnet):
    y_target = label_idx
    target = Variable(torch.LongTensor([y_target]), requires_grad=False)
    
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, target)
    resnet.zero_grad()
    loss_cal.backward(retain_graph=True)
 
    
    eps = 0.90
    grad = torch.sign(img_var.grad.data)                
    adversarial = img_var.data - eps * grad           
    output_adv = resnet.forward(Variable(adversarial))  
    x_adv = labels[torch.max(output_adv.data, 1)[1][0]]    
    _, index1 = torch.max(output_adv, 1)
    op_adv_prob = torch.nn.functional.softmax(output_adv, dim=1)[0] * 100
    adv_percentage = round(op_adv_prob[index1[0]].item(), 4)

    return [x_adv, adv_percentage]
  
    
#fast gradient adversarial example
def fast_gradient(label_idx, labels, output, img_var, resnet):
    y_target = label_idx
    target = Variable(torch.LongTensor([y_target]), requires_grad=False)

    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, target)
    loss_cal.backward(retain_graph=True) 

#higher epsilon value = more preturbation in the image = change in class *smaller values change percentage but not class
    eps = 0.60
    grad = torch.sign(img_var.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
    adversarial = img_var.data + eps * grad          #find adv example using formula 
    output_adv = resnet.forward(Variable(adversarial))   #perform a forward pass on adv example
    x_adv = labels[torch.max(output_adv.data, 1)[1][0]]    #classify the adv example
      #Get probability distribution and find probability (confidence) of a predicted class
    _, index1 = torch.max(output_adv, 1)
    op_adv_prob = torch.nn.functional.softmax(output_adv, dim=1)[0] * 100
    adv_percentage = round(op_adv_prob[index1[0]].item(), 4)

    return [x_adv, adv_percentage]


def main():
    resnet  = models.resnet50(pretrained=True)
    img = Image.open("/home/katelynn/Downloads/__MACOSX/img_4_v1.png").convert('RGB')
    resnet.eval()
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

    output = resnet.forward(img_var)
    label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element
   
    with open('/home/katelynn/Desktop/imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    img_out = labels[label_idx]
    print(img_out)

#calculates probability
    _, index = torch.max(output, 1)
    prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
    percentage = round(prob[index[0]].item(), 4)
    print(percentage)
    #runs the fast gradient sign method
    adv = fast_gradient(label_idx, labels, output, img_var, resnet)
    for i in range(0, len(adv)):
        print(adv[i])
    #runs the one-step target class method
    
    adv_os = one_step_target(label_idx, labels, output, img_var, resnet)
    for i in range(0, len(adv_os)):
        print(adv_os[i])

    #runs the basic iterative method
    adv_bi = basic_iterative(label_idx, labels, output, img_var, img_tensor, resnet)
    for i in range(0, len(adv_bi)):
        print(adv_bi[i])

    adv_it = iterative_target(label_idx, labels, output, img_var, img_tensor, resnet)
    for i in range(0, len(adv_it)):
        print(adv_it[i])
                
main()



    



