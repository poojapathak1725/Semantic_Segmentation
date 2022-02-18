import torch
import torch.nn as nn
from torchvision import models

class OGNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.maxpool = nn.MaxPool2d(kernel_size=4, padding=1, stride=1)
        self.enc1 = self.encoder_block(3,32)
        self.enc2 = self.encoder_block(32,64)
        self.enc3 = self.encoder_block(64,128)
        self.enc4 = self.encoder_block(128,256)
        self.dec1 = self.decoder_block(256,128)
        self.dec2 = self.decoder_block(128,64)
        self.dec3 = self.decoder_block(64,32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=4, padding=1, dilation=1)

    def encoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return block

    def decoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=(2,2)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, padding=1, stride=1, output_padding=1, dilation=(4,4))
        )
        return block

    def forward(self,x):
        x1 = self.maxpool(self.enc1(x))
        x2 = self.maxpool(self.enc2(x1))
        x3 = self.maxpool(self.enc3(x2))
        encoder_output = self.maxpool(self.enc4(x3))

        y1 = self.dec1(encoder_output)
        y2 = self.dec2(y1)
        decoder_output = self.dec3(y2)

        score = self.classifier(decoder_output)                   

        return score