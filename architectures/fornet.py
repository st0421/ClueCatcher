from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from . import externals
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
"""
Feature Extractor
"""


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EfficientNetGenIQA(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGenIQA, self).__init__()
        self.dct_module = DCT_2D()
        self.idct_module  = IDCT_2D()
        self.efficientnet = EfficientNet.from_pretrained(model)
        self.efficientnet2 = EfficientNet.from_pretrained(model)
        self.efficientnet3 = EfficientNet.from_pretrained(model)
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.efficientnet._conv_head.out_channels,256)
        self.classifier = nn.Linear(256+1, 1)
#        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc
        del self.efficientnet2._fc
        del self.efficientnet3._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def features2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet2.extract_features(x)
        x = self.efficientnet2._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def features3(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet3.extract_features(x)
        x = self.efficientnet3._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x,iqa):
            dct = self.dct_module(x)
            tc,th,tw = dct.size()
            mask = torch.ones((th, tw), dtype=torch.int64, device = torch.device('cuda:0'))
            diagonal = tw-40

            ## lf, hf 나누기
            lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
            hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1

            lf_mask = lf_mask.expand(x.size())
            hf_mask = hf_mask.expand(x.size())

            dhf = x * hf_mask
            dlf = x * lf_mask
           
            hf_x = self.idct_module(dhf)
            lf_x = self.idct_module(dlf)

            hf_x = hf_x.permute(1,2,0).cpu().numpy()
            lf_x = lf_x.permute(1,2,0).cpu().numpy()       

            x = self.features(x)        #batch_size * 1792
            hf_x = self.features2(hf_x)
            lf_x = self.features3(lf_x)
        
            x = self.efficientnet._dropout(x)
            hf_x = self.efficientnet2._dropout(hf_x)
            lf_x = self.efficientnet3._dropout(lf_x)
            
            heads=[]
            heads.append(x)
            heads.append(hf_x)
            heads.append(lf_x)
            heads = torch.stack(heads).permute([1,0,2])
            heads = F.log_softmax(heads,dim=1)
            
            x = self.fc(heads.sum(dim=1))
            x = torch.cat((x,iqa),dim=1)
            x = self.classifier(x)
            return x
    
class ClueCatcher(EfficientNetGenIQA):
    def __init__(self):
        super(ClueCatcher, self).__init__(model='efficientnet-b4')



class DCT_2D(nn.Module):
    def __init__(self):
        super(DCT_2D, self).__init__()

    def dct_3d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        X3 = self.dct(X2.transpose(-1, -3), norm=norm)
        return X3.transpose(-1, -3).transpose(-1, -2)
   
   
    def dt_2d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def _rfft(self, x, signal_ndim=1, onesided=True):
        odd_shape1 = (x.shape[1] % 2 != 0)
        x = torch.fft.rfft(x)
        x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
        if onesided == False:
            _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
            _x[:,:,1] = -1 * _x[:,:,1]
            x = torch.cat([x, _x], dim=1)
        return x

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)
        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
        Vc = self._rfft(v, 1, onesided=False)
        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def forward(self, x):
        x = self.dct_3d(x)

        return x

class IDCT_2D(nn.Module):
    def __init__(self):
        super(IDCT_2D, self).__init__()

    def idct_3d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        x3 = self.idct(x2.transpose(-1, -3), norm=norm)
        return x3.transpose(-1, -3).transpose(-1, -2)
   
    def idct_2d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)
   
    def _irfft(self, x, signal_ndim=1, onesided=True):
        if onesided == False:
            res_shape1 = x.shape[1]
            x = x[:,:(x.shape[1] // 2 + 1),:]
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x, n=res_shape1)
        else:
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x)
        return x
   
    def idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = self._irfft(V, 1, onesided=False)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)
   
    def forward(self, x):
        x = self.idct_3d(x)
        return x
