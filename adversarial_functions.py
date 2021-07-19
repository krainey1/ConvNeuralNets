#import required libs
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time 


#Iterative Target Class method
def iterative_target(label_idx, labels, output, img_var, image_tensor, resnet):
    y_true = label_idx
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)

    eps = 0.02
    
    num_steps = 5
    alpha = 0.025

    img_var.data = image_tensor
        
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

#Basic Iterative Method
def basic_iterative(label_idx, labels, output, img_var, image_tensor, resnet):
    y_true = label_idx
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)
    
    eps = 0.02
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
    
    resnet.zero_grad()
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, target)
    loss_cal.backward(retain_graph=True)
 
    
    eps = 0.75
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
    eps = 0.02
    grad = torch.sign(img_var.grad.data)                #calculate the sign of gradient of the loss func (with respect to input X) (adv)
    adversarial = img_var.data + eps * grad          #find adv example using formula 
    output_adv = resnet.forward(Variable(adversarial))   #perform a forward pass on adv example
    x_adv = labels[torch.max(output_adv.data, 1)[1][0]]    #classify the adv example
      #Get probability distribution and find probability (confidence) of a predicted class
    _, index1 = torch.max(output_adv, 1)
    op_adv_prob = torch.nn.functional.softmax(output_adv, dim=1)[0] * 100
    adv_percentage = round(op_adv_prob[index1[0]].item(), 4)

    return [x_adv, adv_percentage]


#def addlabels(x,y):
   #for i in range(len(x)):
       #plt.text(i,y[i],y[i]) 


def main():
    resnet = models.resnet50(pretrained=True)
    #img = Image.open("/home/katelynn/Downloads/__MACOSX/img_7_v1.png").convert('RGB')
    #aengus jank code 

    #Change file path here to where images.txt is being stored
    with open('/home/katelynn/Desktop/Convnets-and-adversarial-functions/images.txt') as file:
        for line in file:
            line = line.replace("\n", "")
            print(line)
            #adjust file to where the sample image folder is located
            img = Image.open(f"/home/katelynn/Desktop/Convnets-and-adversarial-functions/katelynnn_sample_images/{line}").convert('RGB')
    
            resnet.eval()
            preprocess = transforms.Compose([
                 #transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]
                 )])

            img_tensor = preprocess(img)
            #img_tensor = transforms.functional.adjust_brightness(img_tensor, 2)
            img_tensor = img_tensor.unsqueeze(0)
            img_var = Variable(img_tensor, requires_grad=True)

            output = resnet.forward(img_var)
            label_idx = torch.max(output.data, 1)[1][0]   #get an index(class number) of a largest element


            #adjust file path to where imagenet_classes.txt is located 
            with open('/home/katelynn/Desktop/Convnets-and-adversarial-functions/imagenet_classes.txt') as f:
                labels = [line.strip() for line in f.readlines()]
                img_out = labels[label_idx]
               # print("Original Confidence:")
               # print(img_out)
                 
         #calculates probability
            _, index = torch.max(output, 1)
            prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
            percentage = round(prob[index[0]].item(), 4)
           # print(percentage)
            #runs the fast gradient sign method
           # print("Fast Gradient: ")
            adv = fast_gradient(label_idx, labels, output, img_var, resnet)
           # for i in range(0, len(adv)):
           #      print(adv[i]) 
                #runs the one-step target class method
           # print("One-step target class: ")
            adv_os = one_step_target(label_idx, labels, output, img_var, resnet)
            #for i in range(0, len(adv_os)):
                #print(adv_os[i])

             #runs the basic iterative method
           # print("Basic Iterative:")
            adv_bi = basic_iterative(label_idx, labels, output, img_var, img_tensor, resnet)
           # for i in range(0, len(adv_bi)):
                #print(adv_bi[i])

             #runs iterative target class method
           # print("Iterative Target Class:")
            adv_it = iterative_target(label_idx, labels, output, img_var, img_tensor, resnet)
            #for i in range(0, len(adv_it)):
                #print(adv_it[i])

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111)
            classes = [f"OG Confidence: \n {img_out}", f"Fast Gradient: \n {adv[0]}", f"One-Step Target: \n {adv_os[0]}", f"Basic Iterative: \n {adv_bi[0]}", f"Iterative Target: \n {adv_it[0]}"]
            probabilities = [percentage, adv[1], adv_os[1], adv_bi[1], adv_it[1]]
            plt.barh(classes, probabilities)
            for index, value in enumerate(probabilities):
                plt.text(value, index, str(value))
            plt.title(f"{line} Class probabilities")
            fig.autofmt_xdate()
            #adjust where the images will be saved to
            fig.savefig(f"test_folder/{line}", bbox_inches='tight')
            time.sleep(10)
            #plt.show()

main()



