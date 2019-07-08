import torch
import torch.nn as nn

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

def double_conv(in_channels, out_channels, batch_norm=False):
    
    seq = None

    if (batch_norm):
        seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    return seq  

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class SilNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64, batch_norm=True)	
        self.dconv_down2 = double_conv(64, 128, batch_norm=True)
        self.dconv_down3 = double_conv(128, 256, batch_norm=True)     

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up2 = double_conv(128 + 128, 128, batch_norm=True)
        self.dconv_up1 = double_conv(128 + 64, 64, batch_norm=True)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)		
        x = self.maxpool(conv1)			

        conv2 = self.dconv_down2(x)		
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)		
        
        x = self.upsample(x)  
      
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
