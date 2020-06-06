import torch
import torch.nn as nn


class generatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim1 = 12
        self.dim2 = 24
        self.dim3 = 48
        self.dim4 = 96
        self.dim5 = 192

  # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.dim1, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim1),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim1, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim1),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # 256 -> 128
        self.pool0 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim1, kernel_size=3, padding=1, stride=2),
        )
        
        self.enc_conv1 =  nn.Sequential(
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim2, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.dim2, out_channels=self.dim2, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # 128 -> 64
        self.pool1 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(in_channels=self.dim2, out_channels=self.dim2, kernel_size=3, padding=1, stride=2),
        )
        
        self.enc_conv2 =  nn.Sequential(
            nn.Conv2d(in_channels=self.dim2, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),  
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),  
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),  
        )
        
        # 64 -> 32
        self.pool2 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim3, kernel_size=3, padding=1, stride=2),
        )
        
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim4, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(in_channels=self.dim4, out_channels=self.dim4, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.dim4, out_channels=self.dim4, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # 32 -> 16
        self.pool3 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(in_channels=self.dim4, out_channels=self.dim4, kernel_size=3, padding=1, stride=2),
        )

        # bottleneck
        self.bottleneck_conv =  nn.Sequential(
            nn.Conv2d(in_channels=self.dim4, out_channels=self.dim5, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim5),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(in_channels=self.dim5, out_channels=self.dim5, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim5),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),                       
        )

        # decoder (upsampling)
        # 16 -> 32
        self.upsample0 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=self.dim5, out_channels=self.dim5, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.dim5, out_channels=self.dim4, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.dec_conv0 = nn.Sequential(            
            nn.Conv2d(in_channels=self.dim4 * 2, out_channels=self.dim4, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.dim4, out_channels=self.dim4, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # 32 -> 64
        self.upsample1 =  nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=self.dim4, out_channels=self.dim4, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.dim4, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        self.dec_conv1 = nn.Sequential(            
            nn.Conv2d(in_channels=self.dim3 * 2, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # 64 -> 128
        self.upsample2 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=self.dim3, out_channels=self.dim3, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim2, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )  
        
        
        self.dec_conv2 = nn.Sequential(            
            nn.Conv2d(in_channels=self.dim2 * 2, out_channels=self.dim2, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # 128 -> 256
        self.upsample3 =   nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=self.dim2, out_channels=self.dim2, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.dim2, out_channels=self.dim1, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim1),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True), 
        )  
        
        self.dec_conv3 = nn.Sequential(            
            nn.Conv2d(in_channels=self.dim1 * 2, out_channels=3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(3),
            #torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),            
        )


    def forward(self, x):
        # encoder
        x = self.enc_conv0(x)
        e0 = self.pool0(x)
        
        e1p = self.enc_conv1(e0)
        e1 = self.pool1(e1p)
        
        e2p = self.enc_conv2(e1)
        e2 = self.pool2(e2p)
        
        e3p = self.enc_conv3(e2)
        e3 = self.pool3(e3p)

        # bottleneck
        b = self.bottleneck_conv(e3)       
        
        # decoder
        d0 = self.upsample0(b)
        d0 = torch.cat((e3p,d0), 1)
        d0 = self.dec_conv0(d0)
        
        d1 = self.upsample1(d0)
        d1 = torch.cat((e2p,d1), 1)
        d1 = self.dec_conv1(d1)
        
        d2 = self.upsample2(d1)
        d2 = torch.cat((e1p,d2), 1)
        d2 = self.dec_conv2(d2)
        
        d3 = self.upsample3(d2)        
        d3 = torch.cat((x,d3), 1)
        d3 = self.dec_conv3(d3)
        return d3

class discriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()       
        self.dim1 = 12
        self.dim2 = 24
        self.dim3 = 48
        self.dim4 = 96
        self.dim5 = 64
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=self.dim1, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim1),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True), 
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim1, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim1),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        
        self.enc_conv1 =  nn.Sequential(
            nn.Conv2d(in_channels=self.dim1, out_channels=self.dim2, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=self.dim2, out_channels=self.dim2, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
         
        self.enc_conv2 =  nn.Sequential(
            nn.Conv2d(in_channels=self.dim2, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),  
            nn.Conv2d(in_channels=self.dim3, out_channels=self.dim3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(self.dim3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),  
            nn.Conv2d(in_channels=self.dim3, out_channels=3, kernel_size=3, padding=1),            
            nn.BatchNorm2d(3),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),  
        )
        

    def forward(self, x):
        x = self.enc_conv0(x)        
        x = self.enc_conv1(x)        
        x = self.enc_conv2(x)        
        #x = nn.torch.sigmoid(x)
        
        return x