import torch
import torch.nn as nn
import cosmap_unet as unet


class CosMaP_FCN5(nn.Module):
    def __init__(self, use_res = True, out1 = 64, out2 = 256, out3 = 512, out4 = 1024, input = 41, output = 3):
        super().__init__()
        self.l1 = unet.Conv2DResBlock(input, out1) if use_res else unet.Conv2DBlock(input, out1)
        self.l2 = unet.Conv2DResBlock(out1, out2) if use_res else unet.Conv2DBlock(out1, out2)
        self.b1 = nn.BatchNorm2d(out2)
        self.l3 = unet.Conv2DResBlock(out2 + input, out3) if use_res else unet.Conv2DBlock(out2 + input, out3)
        self.l4 = unet.Conv2DResBlock(out3 + out1, out4) if use_res else unet.Conv2DBlock(out3 + out1, out4)
        self.b2 = nn.BatchNorm2d(out4)
        self.l5 = unet.Conv2DBlock(out4 + out2 + 1, output, 1, last_block=True)

    #На вход 41 слой...
    def forward(self, x):
        cosmap = x[:, 0, :, :].unsqueeze(1)
        out1 = self.l1(x)
        out2 = self.b1(self.l2(out1))
        out3 = self.l3(torch.cat([out2, x], 1))
        out4 = self.b2(self.l4(torch.cat([out3, out1], 1)))
        return self.l5(torch.cat([out4, out2, cosmap], 1))

    #Логгирование весов
    def log_weights(self, logger, i):
        logger.add_histogram('Layer1', torch.cat([torch.flatten(p) for p in self.l1.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer2', torch.cat([torch.flatten(p) for p in self.l2.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer3', torch.cat([torch.flatten(p) for p in self.l3.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer4', torch.cat([torch.flatten(p) for p in self.l4.parameters() if p.requires_grad]), i)
        logger.add_histogram('Layer5', torch.cat([torch.flatten(p) for p in self.l5.parameters() if p.requires_grad]), i)


#Версия, где много BatchNorm... (НЕ ОЧЕНЬ)
class Conv2DResBlockBN(unet.Conv2DResBlock):
    def __init__(self, input, output, kernel = 3, dilation = 1):
        super().__init__(input, output, kernel, dilation)
        self.BN = nn.BatchNorm2d(output)

    def forward(self, x):
        return self.BN(super().forward(x))

class CosMaP_FCN5_BN(nn.Module):
    def __init__(self, out1 = 64, out2 = 256, out3 = 512, out4 = 1024):
        super().__init__()
        self.l1 = Conv2DResBlockBN(41, out1)
        self.l2 = Conv2DResBlockBN(out1, out2)
        self.l3 = Conv2DResBlockBN(out2 + 41, out3)
        self.l4 = Conv2DResBlockBN(out3 + out1, out4)
        self.l5 = unet.Conv2DBlock(out4 + out2, 3, 1, last_block=True) # + 1

#Ctrl-C -> Ctrl-V
    def forward(self, x):
        #cosmap = x[:, 0, :, :].unsqueeze(1)
        out1 = self.l1(x)
        out2 = self.l2(out1)
        out3 = self.l3(torch.cat([out2, x], 1))
        out4 = self.l4(torch.cat([out3, out1], 1))
        return self.l5(torch.cat([out4, out2], 1)) #, cosmap

    #Логгирование весов
    def log_weights(self, logger):
        logger.add_histogram('Layer1', torch.cat([torch.flatten(p) for p in self.l1.parameters() if p.requires_grad]))
        logger.add_histogram('Layer2', torch.cat([torch.flatten(p) for p in self.l2.parameters() if p.requires_grad]))
        logger.add_histogram('Layer3', torch.cat([torch.flatten(p) for p in self.l3.parameters() if p.requires_grad]))
        logger.add_histogram('Layer4', torch.cat([torch.flatten(p) for p in self.l4.parameters() if p.requires_grad]))
        logger.add_histogram('Layer5', torch.cat([torch.flatten(p) for p in self.l5.parameters() if p.requires_grad]))



####COUNT PARAMETERS####
from deephd_model import count_model_parameters

if __name__ == '__main__':
    net = CosMaP_FCN5(out4 = 512)
    print(net)
    print(count_model_parameters(net))