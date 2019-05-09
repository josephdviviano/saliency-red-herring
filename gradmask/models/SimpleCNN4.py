import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.register as register


@register.setmodelname("SimpleCNN4")
class CNN(nn.Module):
    def __init__(self, img_size, flat_layer=2, num_class=2):
        super(CNN, self).__init__()
        
        self.img_size = img_size
        flat_layer = flat_layer * 100
        self.all_activations = []
        
        self.ref = []
        
        self.conv1 = nn.Conv2d(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                     ) 
        self.relu1 = nn.ReLU()
        
        #output_size = self.outputSize(img_size, kernel_size=3, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(
                        in_channels=16,
                        out_channels=8,
                        kernel_size=6,
                        stride=1,
                        padding=0,
                    )
        self.relu2 = nn.ReLU()
        
        #output_size = self.outputSize(output_size, kernel_size=3, stride=1, padding=0)
        
        self.conv3 = nn.Conv2d(
                        in_channels=8,
                        out_channels=8,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                    )
        self.relu3 = nn.ReLU()
        
        #output_size = self.outputSize(output_size, kernel_size=3, stride=1, padding=0)
        
        self.conv4 = nn.Conv2d(
                        in_channels=8,
                        out_channels=4,
                        kernel_size=16,
                        stride=1,
                        padding=0,
                    )
        self.relu4 = nn.ReLU()
        
        #output_size = self.outputSize(output_size, kernel_size=3, stride=2, padding=0)

        # self.fc1 = nn.Linear(8 * output_size * output_size, flat_layer)
        self.out = nn.Linear(64, num_class)
        #self.relu5 = nn.ReLU()
        
        #self.out = nn.Linear(flat_layer, num_class)
    
    def outputSize(self, in_size, kernel_size, stride, padding):

        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

        return(output)

    def forward(self, x):
        # reset so we only get the activations for this batch
        self.all_activations = []
        
        # print("input: ", x.shape, "img_size: ", self.img_size)
        x = self.conv1(x)
        self.all_activations.append(x)
                                     
        x = self.relu1(x)
        self.all_activations.append(x)
        
        x = self.conv2(x)
        self.all_activations.append(x)
        
        x = self.relu2(x)
        self.all_activations.append(x)
        
        x = self.conv3(x)
        self.all_activations.append(x)
        
        x = self.relu3(x)
        self.all_activations.append(x)
        
        x = self.conv4(x)
        self.all_activations.append(x)
        
        x = self.relu4(x)
#         self.all_activations.append(x)
        # print("before view: ", x.shape)
        x = x.view(x.size(0), -1) # collapses shape to [batch, channels*height*width]
        # print("after view: ", x.shape)
        #x = self.fc1(x)
        
        #self.all_activations.append(x)

        #x = self.relu5(x)
        # print("after fc1: ", x.shape)
        #self.all_activations.append(x)

        output = self.out(x)
        return output, x