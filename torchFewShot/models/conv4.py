from torch import nn


class ConvNet(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2),)
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(2),)
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),)
        #                nn.MaxPool2d(2),)
        self.nFeat = 512
        # self.fc = nn.Linear(1600, num_classes)

    def forward(self,x, return_feat=False, return_both=False):
        """x: bs*3*84*84 """
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0), -1)
        return out
        #if return_feat:
        #    return out
        #result = self.fc(out)
        #if return_both:
        #   return out, result
        #return result


def conv4():
    return ConvNet()
