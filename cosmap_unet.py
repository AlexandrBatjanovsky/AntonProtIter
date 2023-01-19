#Напишем свою сеть, схожую с U-Net
#Не будем использовать код new_model/deephd_model - напишем свой

import torch
from torch import nn
#import additional_modules as add

#2D свёртка + активация
class Conv2DBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, last_block = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = input,
                              out_channels = output,
                              kernel_size = kernel,
                              padding = 'same',
                              padding_mode = 'circular')
        self.activation = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        return self.activation(self.conv(x))

#1D свёртка + активация
class Conv1DBlock(nn.Module):
    def __init__(self, input, output, kernel = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels = input,
                              out_channels = output,
                              kernel_size = kernel,
                              padding = 'same',
                              padding_mode = 'reflect')
        self.activation = nn.ELU(inplace = True)
     
    def forward(self, x):
        return self.activation(self.conv(x))

#Блок 2D кодировки
class EncoderBlock(nn.Sequential):
    def __init__(self, input, output):
        body = [Conv2DBlock(input, input),
                Conv2DBlock(input, output),
                Conv2DBlock(output, output)]
        super().__init__(*body)

#Блок 1D кодировки
class Encoder1DBlock(nn.Sequential):
    def __init__(self, input, middle, output):
        body = [Conv1DBlock(input, middle),
                Conv1DBlock(middle, middle),
                Conv1DBlock(middle, output)]
        super().__init__(*body)

#Кодировщик
class Encoder(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc1d = Encoder1DBlock(21, 32, 64)
        self.enc3 = EncoderBlock(256, output)

    def forward(self, cosma, amino1, amino2):
        m, n = cosma.shape[2:4]
        #Кодировка матрицы косинусов
        self.inp1 = cosma
        self.out1 = self.enc1(cosma)
        out2 = self.enc2(self.out1)
        #Кодировка аминокислотной последовательности
        out1d_1 = self.enc1d(amino1)
        out1d_1 = torch.unsqueeze(out1d_1, 2)
        out1d_1 = out1d_1.expand(-1, -1, m, -1) #amino1 - 21 x m - error, should be 21 x n
        out1d_2 = self.enc1d(amino2)
        out1d_2 = torch.unsqueeze(out1d_2, 3)
        out1d_2 = out1d_2.expand(-1, -1, -1, n) #amino2 - 21 x n
        #Совместная кодировка
        inp3 = torch.cat([out2, out1d_1, out1d_2], dim = 1)
        return self.enc3(inp3)

#Блок декодировщика
class DecoderBlock(nn.Sequential):
    def __init__(self, input, skip, output, last_block = False):
        body = [Conv2DBlock(input + skip, skip + input, 1),
                Conv2DBlock(skip + input, output, 1, last_block)]
        super().__init__(*body)

#Декодировщик
class Decoder(nn.Module):
    def __init__(self, input, output, use_batch_norm = False):
        super().__init__()
        self.dec1 = DecoderBlock(input, 64, 64)
        #
        self.norm = nn.BatchNorm2d(64, affine=False) if use_batch_norm else nn.Identity()
        #
        self.dec2 = DecoderBlock(64, 1, output, True)

    def forward(self, x, encoder):
        inp1 = torch.cat([x, encoder.out1], dim = 1)
        out1 = self.dec1(inp1)
        #
        out1 = self.norm(out1)
        #
        inp2 = torch.cat([out1, encoder.inp1], dim = 1)
        return self.dec2(inp2)

#Узкое место
class Bottleneck(nn.Sequential):
    def __init__(self, input, output, use_mean = False):
        body = [Conv2DBlock(input, 8 * output),
                Conv2DBlock(8 * output, 4 * output),
                Conv2DBlock(4 * output, 2 * output),
                Conv2DBlock(2 * output, output),
                #add.MeanLayer() if use_mean else
                nn.BatchNorm2d(output)]
        super().__init__(*body)








#############
#Новые блоки#
#############

#ResNet блоки

class Conv2DResBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, dilation = 1, last_block = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = input,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'circular',
                               dilation = dilation)
        self.conv2 = nn.Conv2d(in_channels = output,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'circular',
                               dilation = dilation)
        self.activation1 = nn.ReLU(inplace = False) #False!
        self.activation2 = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        mid = self.conv1(x)
        return self.activation2(mid + self.conv2(self.activation1(mid)))

class Conv1DResBlock(nn.Module):
    def __init__(self, input, output, kernel = 3, last_block = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels = input,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'reflect')
        self.conv2 = nn.Conv1d(in_channels = output,
                               out_channels = output,
                               kernel_size = kernel,
                               padding = 'same',
                               padding_mode = 'reflect')
        self.activation1 = nn.ReLU(inplace = False) #False!
        self.activation2 = nn.Identity() if last_block else nn.LeakyReLU(negative_slope = 0.2, inplace = True)
    
    def forward(self, x):
        mid = self.conv1(x)
        return self.activation2(mid + self.conv2(self.activation1(mid)))

class Encoder1(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.enc1 = Conv2DResBlock(1, 32)
        self.enc2 = Conv2DResBlock(33, 128)
        self.e1d1 = Conv1DResBlock(21, 64)
        self.enc3 = Conv2DResBlock(256, output)
    
    def forward(self, x, a1, a2):
        n = x.size(dim = 2)
        m = x.size(dim = 3)
        self.out1 = self.enc1(x)
        self.out2 = self.enc2(torch.cat([self.out1, x], dim = 1))
        o1d1 = self.e1d1(a1).unsqueeze(2).expand(-1, -1, n, -1)
        o1d2 = self.e1d1(a2).unsqueeze(3).expand(-1, -1, -1, m)
        return self.enc3(torch.cat([self.out2, o1d1, o1d2], dim = 1))

# class Bottleneck1(nn.Sequential):
#     def __init__(self, input, output):
#         body = [Conv2DResBlock(input, 128, dilation = 3),
#                 nn.BatchNorm2d(128),
#                 Conv2DResBlock(128, output, dilation = 3, last_block = True),
#                 add.MeanLayer()]
#         super().__init__(*body)

class DecoderBlock1(nn.Sequential):
    def __init__(self, input):
        body = [Conv2DResBlock(input, 8, 1),
                Conv2DBlock(8, 1, 1, last_block = True)]
        super().__init__(*body)

class Decoder1(nn.Module):
    def __init__(self, input):
        super().__init__()
        self.dec1 = Conv2DResBlock(input + 128, 64, 1)
        self.dec_csm = DecoderBlock1(64)
        self.dec_cvl = DecoderBlock1(48)
        self.dec_cvr = DecoderBlock1(48)

    def forward(self, x, encoder):
        out1 = self.dec1(torch.cat([x, encoder.out2], dim = 1)).split([32, 16, 16], dim = 1)
        csm = self.dec_csm(torch.cat([out1[0], encoder.out1], dim = 1))
        cvl = self.dec_cvl(torch.cat([out1[1], encoder.out1], dim = 1))
        cvr = self.dec_cvr(torch.cat([out1[2], encoder.out1], dim = 1))
        return torch.cat([csm, cvl, cvr], dim = 1)
###################








#Сетошка
class CosMaP_UNet(nn.Module):
    def __init__(self, lastencode = 256, bottleout = 32, output = 2, ver = 0):
        super().__init__()
        self.encoder = Encoder(lastencode) if ver != 1 else Encoder1(lastencode)
        self.bottleneck = Bottleneck(lastencode, bottleout, ver == 2) #if ver != 1 else Bottleneck1(lastencode, bottleout)
        self.decoder = Decoder(bottleout, output, ver == 2) if ver != 1 else Decoder1(bottleout)
        #WEIHGTS INIT
        for m in self.modules():
            if m is nn.Conv2d or m is nn.Conv1d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif m is nn.BatchNorm2d:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #############

    def forward(self, cosma, amino1, amino2):
        return self.decoder(self.bottleneck(self.encoder(cosma, amino1, amino2)), self.encoder)







####################
#Выделение векторов#
####################

# class CosMaP_Extractor(nn.Module):
#     def __init__(self, return_vec1 = True):
#         super().__init__()
#         #self.predictor = predictor
#         self.matrix_extractor = add.MatrixExtractor()
#         #self.return_vec1 = return_vec1
#         self.q_refiner = add.QRefiner(not return_vec1, return_vec1)
#         self.cv_extractor = add.CVExtractor()
        
#     def forward(self, x, t1, t2):
#         #x = self.predictor(cosma, amino1, amino2)
#         return (self.q_refiner(self.matrix_extractor(x[:, 0], t1, t2)),
#                 self.cv_extractor(x[:, 1], t1))

# class TwoVectorLoss(nn.Module):
#     def __init__(self, in_degrees = False, l2 = True, v1_coef = 0.5, cv_coef = 0.5):
#         super().__init__()
#         self.in_degrees = in_degrees
#         self.l2 = l2
#         self.v1_coef = v1_coef
#         self.cv_coef = cv_coef
#         self.v1_loss = add.CosineLoss(in_degrees, True)
#         self.cv_loss = add.CosineLoss(in_degrees, False)

#     def get_errors(self, x, y):
#         return (self.v1_loss(x[0], y[0]), self.cv_loss(x[1], y[1]))

#     def forward(self, x, y):
#         errs = self.get_errors(x, y)
#         if self.l2:
#             result = self.v1_coef * errs[0] ** 2 + self.cv_coef * errs[1] ** 2
#         else:
#             result = self.v1_coef * errs[0] + self.cv_coef * errs[1]
#         return torch.mean(result)

#     def get_angles(self, x, y):
#         errs = self.get_errors(x, y)
#         if self.in_degrees:
#             return errs
#         else:
#             return (add.cos_to_degrees(1 - errs[0]), add.cos_to_degrees(1 - errs[1]))

####COUNT PARAMETERS####
from deephd_model import count_model_parameters

if __name__ == '__main__':
    net = CosMaP_UNet()
    print(net)
    print(count_model_parameters(net))