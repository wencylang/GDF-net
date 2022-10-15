
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vgg as vgg
import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d

__all__ = ['WSDAN']
EPSILON = 1e-12

def knn(features, k):
    inner = -2 * torch.matmul(features.transpose(2, 1), features)
    xx = torch.sum(features ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(features, k=3, idx=None): #features: (B, M, C)
    batch_size = features.size(0)
    num_points = features.size(2) #32
    #features = features.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(features, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = features.size()
    features = features.transpose(2, 1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)
    feature = features.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    features = features.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - features, features), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class GSRR(nn.Module):
    def __init__(self, pool='GAP'):
        super(GSRR, self).__init__()
        self.num_features=768
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_features)
        self.bn2 = nn.BatchNorm2d(self.num_features)
        self.conv1 = nn.Sequential(nn.Conv2d(self.num_features *2, self.num_features, kernel_size=1, bias=False),
                               self.bn1,
                               nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_features * 2, self.num_features, kernel_size=1, bias=False),
                               self.bn2,
                               nn.LeakyReLU(negative_slope=0.2))
    def forward(self, features, attentions):
       B, C, H, W = features.size()
       _, M, AH, AW = attentions.size()
       feature_matrix = []
       for i in range(M):
           AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
           feature_matrix.append(AiF) #(B, M, C)
       feature_matrix = torch.stack(feature_matrix)
       feature_matrix = feature_matrix.transpose(0, 1)
       feature_matrix = feature_matrix.transpose(1, 2)
       features = get_graph_feature(feature_matrix, k=3)
       features = self.conv1(features)
       features1 = features.max(dim=-1, keepdim=False)[0]
       features = get_graph_feature(features1, k=3)
       features = self.conv2(features)
       features2 = features.max(dim=-1, keepdim=False)[0]
       features_fuse = torch.cat((features1, features2), dim=1)
       features_fuse = features_fuse.view(B, -1)
       # sign-sqrt
       features_fuse = torch.sign(features_fuse) * torch.sqrt(torch.abs(features_fuse) + EPSILON)
       # l2 normalization along dimension M and C
       features_fuse = F.normalize(features_fuse, dim=-1)
       return features_fuse


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False,hash_bit_size=48):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)
        self.GSRR = GSRR(pool='GAP')
        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

       
        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)
        self.fc_fuse = nn.Linear(self.M * self.num_features*2, self.num_classes, bias=False)       
        self.fc_fuse_con = nn.Linear(self.M * self.num_features*3, self.num_classes, bias=False)
        # hash layer 
        self.hash_layer = nn.Linear(self.M * self.num_features *3, hash_bit_size)
        # hash layers
        self.fc_hash = nn.Linear(hash_bit_size, self.num_classes, bias=False)
        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)
        feature_fuse = self.GSRR(feature_maps, attention_maps)
        feature_matrix_fuse = torch.cat((feature_matrix,feature_fuse), dim=1)
        # Classification
        # p = self.fc(feature_matrix * 100.)
        #p_fuse = self.fc_fuse(feature_fuse * 100.)
        # p_fuse_con = self.fc_fuse_con(feature_matrix_fuse * 100.)

        # Hash Bit ！！！！CHANGE！！！！
        hash_bit = self.hash_layer(feature_matrix_fuse * 100.)		
        # Classification ！！！！CHANGE！！！！
        p_hash = self.fc_hash(hash_bit)


        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p_hash, feature_matrix_fuse, attention_map,hash_bit

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)

def hash_loss(hash_bit):
    batch_size = hash_bit.size(0)
    tmp = torch.pow(torch.sub(torch.abs(hash_bit), torch.ones(1).cuda()),2)
    quantized_loss = torch.mean(tmp) 

    tmp = torch.where(hash_bit >= 0, torch.ones(1).cuda(), -1*torch.ones(1).cuda())
    tmp = torch.matmul(tmp,torch.ones(hash_bit.size(1),hash_bit.size(0)).cuda())
    balance_loss = torch.mean(torch.pow(tmp, 2))

    loss = balance_loss + quantized_loss

    return loss / batch_size
# class DPSHLoss(torch.nn.Module):
#     def __init__(self, config, bit):
#         super(DPSHLoss, self).__init__()
#         self.q = bit
#         self.U = torch.zeros(config["batch_size"], bit).float().cuda()
#         self.Y = torch.zeros(config["batch_size"], config["n_class"]).float().cuda()

def DPSHLoss(u, y, config):
    u = u / (u.abs() + 1)
    s = y @ y.t()
    norm = y.pow(2).sum(dim=0, keepdim=True).pow(0.5) @ y.pow(2).sum(dim=0, keepdim=True).pow(0.5).t()
    s = s / (norm + 0.00001)

    M = (s > 0.99).float() + (s < 0.01).float()

    inner_product = config["alpha"] * u @ self.U.t()

    log_loss = torch.log(1 + torch.exp(-inner_product.abs())) + inner_product.clamp(min=0) - s * inner_product

    mse_loss = (inner_product + u.size(0)- 2 * s * u.size(0)).pow(2)

    loss1 = (M * log_loss + config["gamma"] * (1 - M) * mse_loss).mean()
    loss2 = config["lambda"] * (u.abs() - 1).abs().mean()

    return loss1 + loss2
