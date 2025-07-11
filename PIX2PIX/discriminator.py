import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding_mode="reflect", bias=False),  #"reflect":- pads with relfections(mirroring) of the input; Usefult to reduce edge artifacts
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)
        
class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, features = [64,128,256,512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels=in_channels, out_channels=feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature
            
        #Add a final_convolution layer to output a single channel instead of 512
        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        
        #We have packed the CNNBlocks into a list; Now we will unpack it 
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, target):
        x = torch.cat([x, target], dim=1)
        return self.model(self.initial(x))
        
        