import torch
import torch.nn as nn
from torchvision import models


class UNet2(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.enc1 = self.encoder_block(3,64)
        self.enc2 = self.encoder_block(64,128)
        self.enc3 = self.encoder_block(128,256)
        self.enc4 = self.encoder_block(256,512)
        self.enc5 = self.encoder_block(512,1024)
        self.dec1 = self.decoder_block(1024,512)
        self.dec2 = self.decoder_block(512*2, 256)
        self.dec3 = self.decoder_block(256*2, 128)
        self.dec4 = self.decoder_block(128*2, 64)
        self.dec5 = self.decoder_block(64*2, 64)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def decoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels*2, kernel_size=3, padding=1, stride=2, output_padding=1, dilation=1),
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def forward(self, x):

        x1 = self.maxpool(self.enc1(x))
        x2 = self.maxpool(self.enc2(x1))
        x3 = self.maxpool(self.enc3(x2))
        x4 = self.maxpool(self.enc4(x3))
        encoder_output = self.maxpool(self.enc5(x4))

        y1 = self.dec1(encoder_output)
        y2 = self.dec2(torch.cat([y1,x4],1))
        y3 = self.dec3(torch.cat([y2,x3],1))
        y4 = self.dec4(torch.cat([y3,x2],1))
        decoder_output = self.dec5(torch.cat([y4,x1],1))

        score = self.classifier(decoder_output)                   

        return score