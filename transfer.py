from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

class TransferNet():
    def __init__(self, n_class):
        self.n_class = n_class

    def createDeepLabModel(self):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        model.classifier = DeepLabHead(2048, num_classes=self.n_class)
        model.train()
        return model