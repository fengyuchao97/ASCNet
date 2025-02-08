import torch
import torch.nn as nn
import math
from einops import rearrange
import torch.nn.functional as F
from .GhostNetv2 import ghostnetv2
from .video_swin_transformer import SwinTransformer3D
from mmcv.cnn import build_norm_layer

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, chunted=4):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn([chunted, dim, shape]))

    def forward(self, x):
        B, _, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x

def shunted(x, chunk=4, dim=-1):
        B, C, H, W = x.shape
        if dim==(-1):
            x = x.reshape(B, C, H, W//chunk, chunk).permute(0,4,1,2,3).mean(dim)
        elif dim==(-2):
            x = x.reshape(B, C, chunk, H//chunk, W).permute(0,2,1,3,4).mean(dim)
        return x

class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim=16, num_heads=8,
                 chunk_number=4,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.chunk_number = chunk_number

        self.to_q_1 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)
        self.to_q_2 = Conv2d_BN(dim, nh_kd//2, 1, norm_cfg=norm_cfg)

        self.to_k_1 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k_2 = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)

        self.to_v_1 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        self.to_v_2 = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)

        self.proj_encode_column_1 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_column_2 = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        
        self.dwconv_1 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        self.dwconv_2 = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv_1 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.pwconv_2 = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
    
    def forward(self, x1, x2, label=None):  
        B, C, H, W = x1.shape

        q_1 = self.to_q_1(x1)
        q_2 = self.to_q_2(x2)
        q = torch.cat([q_1,q_2],dim=1)

        k_1 = self.to_k_1(x1)
        k_2 = self.to_k_2(x2)
        k = torch.abs(k_1-k_2)

        v_1 = self.to_v_1(x1)
        v_2 = self.to_v_2(x2)
        
        # detail enhance
        qkv_1 = torch.cat([q, k, v_1], dim=1)
        qkv_1 = self.act(self.dwconv_1(qkv_1))
        qkv_1 = self.pwconv_1(qkv_1)

        qkv_2 = torch.cat([q, k, v_2], dim=1)
        qkv_2 = self.act(self.dwconv_2(qkv_2))
        qkv_2 = self.pwconv_2(qkv_2)

        # squeeze axial attention
        ## squeeze row
        # B, chunt, C, H, W 
        qrow = self.pos_emb_rowq(shunted(q, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow = self.pos_emb_rowk(shunted(k, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        vrow_1 = shunted(v_1, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        vrow_2 = shunted(v_2, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)


        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)

        xx_row_1 = torch.matmul(attn_row, vrow_1)  # B nH H C
        xx_row_1 = self.proj_encode_row_1(xx_row_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        xx_row_2 = torch.matmul(attn_row, vrow_2)
        xx_row_2 = self.proj_encode_row_2(xx_row_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        ## squeeze column
        qcolumn = self.pos_emb_columnq(shunted(q, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn = self.pos_emb_columnk(shunted(k, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        vcolumn_1 = shunted(v_1, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        vcolumn_2 = shunted(v_2, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)

        xx_column_1 = torch.matmul(attn_column, vcolumn_1)  # B nH W C
        xx_column_1 = self.proj_encode_column_1(xx_column_1.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx_column_2 = torch.matmul(attn_column, vcolumn_2)  # B nH W C
        xx_column_2 = self.proj_encode_column_2(xx_column_2.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)


        xx = torch.abs(xx_row_1-xx_row_2).add(torch.abs(xx_column_1-xx_column_2)).reshape(B, self.dh, H, W)
        xx = torch.abs(v_1-v_2).add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out1 = att * qkv_1
        out2 = att * qkv_2

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)
            loss_res = self.loss_generator(qkv_1*(1-label), qkv_2*(1-label))
            return out1, out2, loss_att, loss_res
        else:
            return out1, out2
    
def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d

def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU

def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)

def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
class DWConv_T(nn.Module):
    def __init__(self, dim=768):
        super(DWConv_T, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """
        x: NHWC tensor
        """
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) #NHWC
        return x

class BasicConv(nn.Module):
    def __init__(
        self, in_ch, out_ch, 
        kernel_size, pad_mode='Zero', 
        bias='auto', norm=False, act=False, 
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(getattr(nn, pad_mode.capitalize()+'Pad2d')(kernel_size//2))
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size,
                stride=1, padding=0,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

class Conv1x1(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 1, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output

class CCAMDec(nn.Module):
    def __init__(self):
        super(CCAMDec,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x,y):
        m_batchsize,C,width ,height = x.size()
        x_reshape =x.view(m_batchsize,C,-1)

        B,K,W,H = y.size()
        y_reshape =y.view(B,K,-1)
        proj_query  = x_reshape 
        proj_key  = y_reshape.permute(0,2,1) 
        energy =  torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy,-1,keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B,K,-1) 
        
        out = torch.bmm(attention,proj_value) 
        out = out.view(m_batchsize,C,width ,height)

        out = x + self.scale*out
        return out

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))
    
class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes=1):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch, out_ch) 
        self.upsample = nn.Sequential(nn.Conv2d(out_ch, out_ch*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out = Conv1x1(out_ch//2, num_classes)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x2 = F.interpolate(x2, size=x1.shape[2:])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        out = self.conv_fuse(x)
        output = self.conv_out(self.upsample(out))
        return out, output

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) #NCHW
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x.permute(0, 2, 3, 1) #NHWC
        return x
            
class ContrastiveAtt_Block(nn.Module):
    def __init__(self, in_channels, drop_path=0.1, chunk_number=4, mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
        super().__init__()     # drop_path=0., mlp_ratio=4, mlp_dwconv=False,

        dim = in_channels #// 2
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing

        self.attn = Sea_Attention(dim, chunk_number=chunk_number) 

        self.pre_norm = pre_norm
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, t1, t2, labels=None):
        # conv pos embedding 
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        # attention & mlp
        if self.pre_norm:
            if labels is not None:
                x1, x2, loss_att, loss_res = self.attn(t1, t2, labels)
            else:
                x1, x2 = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)
            
            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)
        if labels is not None:
            return t1, t2, loss_att, loss_res
        else:
            return t1, t2

class Local_LSKAtt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0_img1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_img1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_img1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2_img1 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze_img1 = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_img1 = nn.Conv2d(dim//2, dim, 1)

        self.conv0_img2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_img2 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1_img2 = nn.Conv2d(dim, dim//2, 1)
        self.conv2_img2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze_img2 = nn.Conv2d(2, 2, 7, padding=3)
        self.conv_img2 = nn.Conv2d(dim//2, dim, 1)

        self.loss_generator = nn.L1Loss()
        # self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2, labels = None):   
        attn1_img1 = self.conv0_img1(x1)
        attn2_img1 = self.conv_spatial_img1(attn1_img1)
        attn1_img1 = self.conv1_img1(attn1_img1)
        attn2_img1 = self.conv2_img1(attn2_img1)
        attn_img1 = torch.cat([attn1_img1, attn2_img1], dim=1)
        avg_attn_img1 = torch.mean(attn_img1, dim=1, keepdim=True)
        max_attn_img1, _ = torch.max(attn_img1, dim=1, keepdim=True)
        agg_img1 = torch.cat([avg_attn_img1, max_attn_img1], dim=1)
        sig_img1 = self.conv_squeeze_img1(agg_img1).sigmoid()

        attn1_img2 = self.conv0_img2(x2)
        attn2_img2 = self.conv_spatial_img2(attn1_img2)
        attn1_img2 = self.conv1_img2(attn1_img2)
        attn2_img2 = self.conv2_img2(attn2_img2)
        attn_img2 = torch.cat([attn1_img2, attn2_img2], dim=1)
        avg_attn_img2 = torch.mean(attn_img2, dim=1, keepdim=True)
        max_attn_img2, _ = torch.max(attn_img2, dim=1, keepdim=True)
        agg_img2 = torch.cat([avg_attn_img2, max_attn_img2], dim=1)
        sig_img2 = self.conv_squeeze_img2(agg_img2).sigmoid()

        # sig_img = torch.abs(sig_img1-sig_img2)
        attn_img1 = attn1_img1 * sig_img1[:,0,:,:].unsqueeze(1) + attn2_img1 * sig_img1[:,1,:,:].unsqueeze(1)
        attn_img1 = self.conv_img1(attn_img1)

        attn_img2 = attn1_img2 * sig_img2[:,0,:,:].unsqueeze(1) + attn2_img2 * sig_img2[:,1,:,:].unsqueeze(1)
        attn_img2 = self.conv_img2(attn_img2)

        att = torch.abs(attn_img1-attn_img2)
        x1 = x1 * att #+ self.scale*x1 # (att + self.scale*attn_img1) 
        x2 = x2 * att #+ self.scale*x2 # (att + self.scale*attn_img2) #att # 

        if labels is not None:
            m_batchsize, C, width, height = x1.size()
            label = F.interpolate(labels, size=(width,height))
            loss_att = self.loss_generator(torch.mean(att,dim=1).sigmoid(),label)
            return x1, x2, loss_att
        else:
            return x1, x2

class Local_Block(nn.Module):
    def __init__(self, in_channels, window_size=8, drop_path=0.1, chunk_number=4, mlp_ratio=3, mlp_dwconv=True, before_attn_dwconv=3, pre_norm=True, norm_layer=nn.BatchNorm2d):
        super().__init__()     # drop_path=0., mlp_ratio=4, mlp_dwconv=False,

        dim = in_channels #//2
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing

        self.attn = Sea_Attention(dim, chunk_number=chunk_number)# Local_LSKAtt(dim)

        self.pre_norm = pre_norm
        self.mlp = FeedForward(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, t1, t2, labels = None):
        # conv pos embedding  3×3卷积，一个残差连接
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)

        # attention & mlp
        loss_att = 0
        if self.pre_norm:
            # x1, x2 = self.attn(t1, t2)
            if labels is not None:
                x1, x2, loss_att, loss_res = self.attn(t1, t2, labels)
            else:
                x1, x2 = self.attn(t1, t2)

            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)

            t1 = t1.permute(0, 2, 3, 1)
            t2 = t2.permute(0, 2, 3, 1)

            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C) 
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        t1 = t1.permute(0, 3, 1, 2)
        t2 = t2.permute(0, 3, 1, 2)

        if labels is not None:
            return t1, t2, loss_att, loss_res
        else:
            return t1, t2

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)
    
class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out

class Fusion(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(Fusion, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y
           
class Align(nn.Module):
    def __init__(self, input_dim, dim, key_dim=16, num_heads=8,
                 chunk_number=4,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN2d', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.chunk_number = chunk_number
        
        self.conv_ccam_b = nn.Sequential(nn.Conv2d(input_dim, dim, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU()) 
        self.ccam_enc = nn.Sequential(nn.Conv2d(dim, dim//16, 1, bias=False),
                                   nn.BatchNorm2d(dim//16),
                                   nn.ReLU()) 
        self.ccam_dec = CCAMDec()
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)

        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16, chunted=self.chunk_number)
        
        self.dwconv = Conv2d_BN(self.dh + 2 * self.nh_kd, 2 * self.nh_kd + self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.nh_kd + self.dh, norm_cfg=norm_cfg)
        
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()
        
        self.loss_generator = nn.L1Loss()
        
    def forward(self, x, label=None):
        # x = self.conv_ccam_b(x)
        ccam_b = self.conv_ccam_b(x)
        ccam_f = self.ccam_enc(ccam_b)
        x = self.ccam_dec(ccam_b,ccam_f)
        
        B, C, H, W = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(shunted(q, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)
        krow = self.pos_emb_rowk(shunted(k, chunk=self.chunk_number, dim=-1)).reshape(B, self.chunk_number, self.num_heads, -1, H)
        vrow = shunted(v, chunk=self.chunk_number, dim=-1).reshape(B, self.chunk_number, self.num_heads, -1, H).permute(0, 1, 2, 4, 3)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)

        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, H//self.chunk_number)).unsqueeze(-1)

        ## squeeze column
        qcolumn = self.pos_emb_columnq(shunted(q, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)
        kcolumn = self.pos_emb_columnk(shunted(k, chunk=self.chunk_number, dim=-2)).reshape(B, self.chunk_number, self.num_heads, -1, W)
        vcolumn = shunted(v, chunk=self.chunk_number, dim=-2).reshape(B, self.chunk_number, self.num_heads, -1, W).permute(0, 1, 2, 4, 3)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)

        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 2, 4, 1, 3).reshape(B, self.dh, self.chunk_number*self.chunk_number, W//self.chunk_number)).unsqueeze(3)

        xx = xx_row.add(xx_column).reshape(B, self.dh, H, W)
        xx = v.add(xx)
        att = self.sigmoid(self.proj(xx))
        
        out = att * qkv

        if label is not None:
            label = F.interpolate(label, size=(H,W))
            loss_att = self.loss_generator(torch.mean(att,dim=1),label)
            return out, loss_att
        else:
            return out
    
import warnings
warnings.filterwarnings("ignore")
class ASCNet(nn.Module): 
    def __init__(self, num_classes=1, normal_init=True, pretrained=False):
        super(ASCNet, self).__init__()
        
        self.video_len = 8 
        self.model = ghostnetv2()
        params=self.model.state_dict() 
        save_model = torch.load('./pretrained/ck_ghostnetv2_10.pth.tar')
        state_dict = {k: v for k, v in save_model.items() if k in params.keys()}
        self.model.load_state_dict(state_dict)
        
        self.global_consrative2 = ContrastiveAtt_Block(192, chunk_number=16) 
        self.global_consrative1 = ContrastiveAtt_Block(80, chunk_number=16) 
        self.local_consrative2 = Local_Block(48, window_size=16, chunk_number=16)
        self.local_consrative1 = Local_Block(32, window_size=16, chunk_number=16)

        self.backbone = SwinTransformer3D()

        self.Translayer2_1 = BasicConv2d(96,64,1)
        self.fam32_1 = Align(112, 64, chunk_number=16) 
        self.Translayer3_1 = BasicConv2d(64,32,1)
        self.fam43_1 = Align(56, 32, chunk_number=16) 

        self.fusion3 = Fusion(192,128,reduction=False) 
        self.fusion2 = Fusion(80,64,reduction=False)
        self.fusion1 = Fusion(48,32,reduction=False)
        self.fusion0 = Fusion(32,16,reduction=False)

        self.decoder1 = DecBlock(128+128+64, 128, num_classes) 
        self.decoder2 = DecBlock(128+64, 64, num_classes)
        self.decoder3 = DecBlock(64+16, 32, num_classes)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        
        self.upsample_pixel = nn.Sequential(nn.Conv2d(32, 32*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        self.conv_out_v = Conv1x1(16, num_classes)

        if normal_init:
            self.init_weights()
    
    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        frames = rearrange(frames, "n l c h w -> n c l h w")
        return frames
    
    def generate_transition_video_tensor(self, frame1, frame2, num_frames=8):
        transition_frames = []

        for t in torch.linspace(0, 1, num_frames):
            weighted_frame1 = frame1 * (1 - t)
            weighted_frame2 = frame2 * t
            blended_frame = weighted_frame1 + weighted_frame2
            transition_frames.append(blended_frame.unsqueeze(0))

        transition_video = torch.cat(transition_frames, dim=0)
        frame = rearrange(transition_video, "l n c h w -> n c l h w")
        return frame

    def forward(self, imgs, labels=None, return_aux=True):
        
        img1 = imgs[:,:,2,:,:]
        img2 = imgs[:,:,3,:,:]
        x, encoder_outputs = self.backbone(imgs)

        if labels is not None:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3, loss_att_swim3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1),labels) #64
            out4, loss_att_swim4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1),labels) #32
        else:
            out2 = self.Translayer2_1(encoder_outputs[2]) #64
            out3 = self.fam32_1(torch.cat([encoder_outputs[1], self.upsamplex2(out2)],1)) #64
            out4 = self.fam43_1(torch.cat([encoder_outputs[0], self.upsamplex2(self.Translayer3_1(out3))],1)) #32

        c0 = self.model.act1(self.model.bn1(self.model.conv_stem(img1)))
        c1 = self.model.blocks[0](c0)
        c2 = self.model.blocks[1](c1) 
        c3 = self.model.blocks[2](c2) 
        c4 = self.model.blocks[3](c3) 
        c5 = self.model.blocks[4](c4) 
        c6 = self.model.blocks[5](c5) 
        c7 = self.model.blocks[6](c6) 

        c0_img2 = self.model.act1(self.model.bn1(self.model.conv_stem(img2)))
        c1_img2 = self.model.blocks[0](c0_img2) 
        c2_img2 = self.model.blocks[1](c1_img2) 
        c3_img2 = self.model.blocks[2](c2_img2) 
        c4_img2 = self.model.blocks[3](c3_img2) 
        c5_img2 = self.model.blocks[4](c4_img2) 
        c6_img2 = self.model.blocks[5](c5_img2) 
        c7_img2 = self.model.blocks[6](c6_img2) 

        if labels is not None:
            cur1_0, cur2_0, loss_local_att1, loss_local_res1 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1),labels) 
            cur1_1, cur2_1, loss_local_att2, loss_local_res2 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1),labels) 
            cur1_2, cur2_2, loss_att2, loss_res2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1),labels) 
            cur1_3, cur2_3, loss_att1, loss_res1 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1),labels) 
            loss_att = loss_att1 + loss_att2 + 0.4*(loss_local_att1 + loss_local_att2) + loss_att_swim3 + loss_att_swim4
            loss_res = loss_res1 + loss_res2 + 0.4*(loss_local_res1 + loss_local_res2)
        else:
            cur1_0, cur2_0 = self.local_consrative1(torch.cat([c0,c1],1), torch.cat([c0_img2,c1_img2],1)) 
            cur1_1, cur2_1 = self.local_consrative2(torch.cat([c2,c3],1), torch.cat([c2_img2,c3_img2],1)) 
            cur1_2, cur2_2 = self.global_consrative1(torch.cat([c4,c5],1), torch.cat([c4_img2,c5_img2],1)) 
            cur1_3, cur2_3 = self.global_consrative2(torch.cat([c6,c7],1), torch.cat([c6_img2,c7_img2],1)) 

        fuse3 = self.fusion3(cur1_3,cur2_3) 
        fuse2 = self.fusion2(cur1_2,cur2_2) 
        fuse1 = self.fusion1(cur1_1,cur2_1) 
        fuse0 = self.fusion0(cur1_0,cur2_0) 
 
        cat3 = torch.cat([fuse3,out2],1) 
        cat2 = torch.cat([fuse2,out3],1) 
        cat1 = torch.cat([fuse1,out4],1) 

        dec1,output_middle2 = self.decoder1(cat2,cat3) 
        dec2,output_middle1 = self.decoder2(cat1,dec1) 
        dec3,output = self.decoder3(fuse0,dec2) 

        if return_aux:
            output_middle2 = F.interpolate(output_middle2, size=output_middle1.shape[2:])
            output_middle1 = F.interpolate(output_middle1, size=output.shape[2:])
            pred_v = self.conv_out_v(self.upsample_pixel(out4))
            pred_v = F.interpolate(pred_v, size=output.shape[2:])

            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            output_middle1 = torch.sigmoid(output_middle1)
            output_middle2 = torch.sigmoid(output_middle2)
    
            pred_v = torch.sigmoid(pred_v)

            if labels is not None:
                return output, output_middle1, output_middle2, pred_v, loss_att, loss_res
            else:
                return output, output_middle1, output_middle2, pred_v
        else:
            output = F.interpolate(output, size=img1.shape[2:])
            output = torch.sigmoid(output)
            if labels is not None:
                return output, loss_att, loss_res
            else:
                return output

    def init_weights(self):
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fusion3.apply(init_weights) 
        self.fusion2.apply(init_weights) 
        self.fusion1.apply(init_weights) 
        self.fusion0.apply(init_weights) 
        
        self.decoder1.apply(init_weights) 
        self.decoder2.apply(init_weights) 
        self.decoder3.apply(init_weights) 
        self.conv_out_v.apply(init_weights) 
