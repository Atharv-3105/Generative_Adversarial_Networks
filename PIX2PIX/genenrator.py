import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout = False, act="relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=4, stride=2, padding=1, bias=False ,padding_mode="reflect")
            if down #It means Conv2d() will act when we are in the Down phase of our UNET like Block
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False), #It means ConvTranspose2d() will act when we are in the Up phase of our UNET like Block
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),   
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, features = 64):
        super().__init__()
        #We will not use BatchNormalization in the first layer
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1,padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) #256----->128
        self.down1 = Block(features, features*2, act="leaky", use_dropout=False)#128----->64
        self.down2 = Block(features*2, features*4, act="leaky", use_dropout=False)#64----->32
        self.down3 = Block(features*4, features*8, act="leaky", use_dropout=False)#32----->16
        self.down4 = Block(features*8, features*8, act="leaky", use_dropout=False)#16----->8
        self.down5 = Block(features*8, features*8, act="leaky", use_dropout=False)#8----->4
        self.down6 = Block(features*8, features*8, act="leaky", use_dropout=False)#4----->2
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        
        self.up1 = Block(features*8, features*8, down=False ,act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False ,act="relu", use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False ,act="relu", use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False ,act="relu", use_dropout=False)
        self.up5 = Block(features*8*2, features*4, down=False ,act="relu", use_dropout=False)
        self.up6 = Block(features*4*2, features*2, down=False ,act="relu", use_dropout=False)
        self.up7 = Block(features*2*2, features, down=False ,act="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=features*2, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), #As we want the pixels values to be between -1 and 1
        )
        
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        bottleneck = self.bottleneck(d7)
        
        #For the Up Phase we need to concatenate the up_convolution with the respective down_convolution
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        
        return self.final_up(torch.cat([up7, d1], dim=1))