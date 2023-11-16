import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt
import pywt


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.0)
            # torch.nn.init.xavier_normal_(self.conv.bias, gain=1)


    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # print(T,self.stride)
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode="nearest"
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs)
            )
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


# attention / transformers
class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the input embedding
        n_head,  # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float("-inf"))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask

class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # dowsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        print("MaskedMHCA's nhead is: ", self.n_head)
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride
        # # # # # # # #thumos14
        # 4  sym2 81.8 78.3 71.9 58.8 44  avg mAP: 67.0   
        # 6  coif1 82.4 78.4 71.4 59.7 44.2  avg mAP: 67.2
        # 62  dmey  82.4 78.6  71.1  59.2  43.7  67.0
        # 4   db2  81.8 78.3 71.9 58.8 44 avg mAP: 67.0
        # # # # # # # #epic-verb
        # sym2    27.3 26.2 24.4 22.4 18.7 avg mAP: 23.8  
        # coif1   27.5 26.7 25.0 22.6 18.9 avg mAP:24.1
        # dmey    26.6 25.6 24.2 22.0 18.4 avg mAP:23.4
        # # # # # # # #epic-noun
        # sym2 26.6 25.2 23.6 21.0 17.3  avg mAP:22.8
        # coif1 26.9 25.6 23.3 20.8 17.0 avg mAP:22.7
        # dmey 26.5 25.3 23.3 20.5 17.6 avg mAP:22.7

        wave_choosed = "db1" # "sym2"  "coif1"  "bior1.1"/"haar"/"db1"  "dmey"
        if self.n_qx_stride > 1:
            # query conv (depthwise)
            stride = self.n_kv_stride
            
            self.query_conv = MaskDWT_1D(
                n_embd=n_embd, wave=wave_choosed, fusion="element_product", stride=stride,
            )  # haar24.0   db1  bior1.1
            self.query_norm = LayerNorm(self.n_embd)

            # key, value conv (depthwise)
            stride = self.n_kv_stride
            self.key_conv = MaskDWT_1D(n_embd=n_embd,wave=wave_choosed,fusion="element_product", stride=stride)
            self.key_norm = LayerNorm(self.n_embd)
            self.value_conv = MaskDWT_1D(n_embd=n_embd,wave=wave_choosed,fusion="element_product", stride=stride)
            self.value_norm = LayerNorm(self.n_embd)
        else:
            # query conv (depthwise)
            stride = 2
            self.query_conv = MaskDWT_1D(n_embd=n_embd,wave=wave_choosed,fusion="element_product", stride=stride,is_stack=True)  # haar24.0   db1  bior1.1
            self.query_norm = LayerNorm(self.n_embd)

            # key, value conv (depthwise)
            self.key_conv = MaskDWT_1D(n_embd=n_embd,wave=wave_choosed,fusion="element_product", stride=stride,is_stack=True)
            self.key_norm = LayerNorm(self.n_embd)
            self.value_conv = MaskDWT_1D(n_embd=n_embd,wave=wave_choosed,fusion="element_product", stride=stride,is_stack=True)
            self.value_norm = LayerNorm(self.n_embd)

        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        padding = kernel_size // 2

        # self.query_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)
        # self.key_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)
        # self.out_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # print("888888888888888888888",x.size())
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        # print("9999999999999999999999",q.size())
        q = self.query_norm(q)
        # qx_mask = qx_mask.transpose(-2, -1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        # kv_mask = kv_mask.transpose(-2, -1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        # q = self.query(self.query_pooling(q))
        # k = self.key(self.key_pooling(k))
        # v = self.value(v)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # print("MaskedMHCA's q.size() is :", q.size())
        # ? calculate attention in channels
        # (B, nh * hs, T'/T'')
        # k = k.view(B, self.n_head, self.n_channels, -1)
        # q = q.view(B, self.n_head, self.n_channels, -1)
        # v = v.view(B, self.n_head, self.n_channels, -1)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        # att = ((q * self.scale).transpose(-2, -1)) @ k
        att = (q * self.scale) @ k.transpose(-2, -1)
        # print("MaskedMHCA's att.size() is :", att.size())

        # prevent q from attending to invalid tokens
        # att = att.masked_fill(torch.logical_not(
        #     kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # ? calculate attention in channels
        # out = att @ (v * kv_mask[:, :, None, :].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        # out = self.out_pooling(out)
        return out, qx_mask

# useless in this project
class MaskedMHCA_Channel(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # dowsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        # print("MaskedMHCA_Channel's nhead is: ", self.n_head)
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.query_norm = LayerNorm(self.n_embd)
        # self.query_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.value_norm = LayerNorm(self.n_embd)

        # self.key_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)
        # self.out_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # qx_mask = qx_mask.transpose(-2, -1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        # kv_mask = kv_mask.transpose(-2, -1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        # q = self.query(self.query_pooling(q))
        # k = self.key(self.key_pooling(k))
        # v = self.value(v)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        # k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # ? calculate attention in channels
        # (B, nh * hs, T'/T'')
        k = k.view(B, self.n_head, self.n_channels, -1)
        q = q.view(B, self.n_head, self.n_channels, -1)
        v = v.view(B, self.n_head, self.n_channels, -1)
        # print("calculate attention in channels:q,k",q.size(),k.size())
        # print("MaskedMHCA_Channel's q.size() is :", q.size())
        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        # att = ((q * self.scale).transpose(-2, -1)) @ k
        att = (q * self.scale) @ k.transpose(-2, -1)
        # print("calculate attention in channels:attention",att.size())

        # prevent q from attending to invalid tokens
        # att = att.masked_fill(torch.logical_not(
        #     kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        # print("calculate attention in channels:attention after softmax",att.size())
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        # out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # ? calculate attention in channels
        out = att @ (v * kv_mask[:, :, None, :].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        # out = self.out_pooling(out)
        return out, qx_mask


class TimeMaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # dowsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        print("TimeMaskedMHCA's nhead is: ", self.n_head)
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.query_norm = LayerNorm(self.n_embd)
        # self.query_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.value_norm = LayerNorm(self.n_embd)

        # self.key_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)
        # self.out_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # qx_mask = qx_mask.transpose(-2, -1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        # kv_mask = kv_mask.transpose(-2, -1)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        # q = self.query(self.query_pooling(q))
        # k = self.key(self.key_pooling(k))
        # v = self.value(v)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # print("MaskedMHCA's q.size() is :", q.size())
        # ? calculate attention in channels
        # (B, nh * hs, T'/T'')
        # k = k.view(B, self.n_head, self.n_channels, -1)
        # q = q.view(B, self.n_head, self.n_channels, -1)
        # v = v.view(B, self.n_head, self.n_channels, -1)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        # att = ((q * self.scale).transpose(-2, -1)) @ k
        att = (q * self.scale) @ k.transpose(-2, -1)
        # print("MaskedMHCA's att.size() is :", att.size())

        # prevent q from attending to invalid tokens
        # att = att.masked_fill(torch.logical_not(
        #     kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # ? calculate attention in channels
        # out = att @ (v * kv_mask[:, :, None, :].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        # out = self.out_pooling(out)
        return out, qx_mask


# Decompose the input into two frequency signals: low-frequency and high-frequency subbands.
class MaskDWT_1D(nn.Module):
    def __init__(
        self, 
        wave, 
        stride, 
        n_embd=None, 
        n_hidden=None,
        n_out=None,
        fusion= 'element_product',
        is_stack=False,
        act_layer=nn.ReLU,
        proj_pdrop=0.0,  # dropout rate for the projection / MLP
        padding=1, 
        dilation=1, 
        bias=False, 
        padding_mode="zeros"
    ):
        super(MaskDWT_1D, self).__init__()
        
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        # print("dec_hi,dec_lo:",dec_hi.size(),dec_lo.size())
        # print("dec_hi:",dec_hi)
        # print("dec_lo:",dec_lo)

        self.register_buffer("dec_hi", dec_hi.unsqueeze(0).unsqueeze(0))
        self.register_buffer("dec_lo", dec_lo.unsqueeze(0).unsqueeze(0))

        self.dec_hi = self.dec_hi.to(dtype=torch.float32)
        self.dec_lo = self.dec_lo.to(dtype=torch.float32)
        # print(self.dec_hi.size())

        self.stride = stride
        # self.padding = padding
        # self.low_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.high_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.is_stack = is_stack
        self.fusion = fusion
        
        # if n_hidden==None:
        #     n_hidden = 2 * n_embd
        # if n_out==None:
        #     n_out = n_embd
            
        # self.conv_low = nn.Sequential(
        #     nn.Conv1d(n_embd, n_hidden, 1),
        #     act_layer(),
        #     nn.Dropout(proj_pdrop, inplace=True),
        #     nn.Conv1d(n_hidden, n_out, 1),
        #     nn.Dropout(proj_pdrop, inplace=True),
        # )
        
        # self.conv_high = nn.Sequential(
        #     nn.Conv1d(n_embd, n_hidden, 1),
        #     act_layer(),
        #     nn.Dropout(proj_pdrop, inplace=True),
        #     nn.Conv1d(n_hidden, n_out, 1),
        #     nn.Dropout(proj_pdrop, inplace=True),
        # )
        
        # self.low_factor = nn.Parameter(torch.tensor(0.5, dtype=torch.float32),requires_grad=True)
        # self.high_factor = nn.Parameter(torch.tensor(0.5, dtype=torch.float32),requires_grad=True)
        # if self.fusion == 'weight_matrix':
        #     pass
            # if is_upsample:
            #     self.wavefusion = WaveFusion(n_embd=n_embd,n_hidden=n_hidden,n_out=n_embd)
            # else:
            #     self.wavefusion = WaveFusion(n_embd=n_embd,n_hidden=n_hidden,n_out=n_embd//self.stride)
        

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # dwt
        x = x.contiguous()
        dim = x.shape[1]
        # print(
        #     "self.dec_lo.expand(dim, -1, -1):", self.dec_lo.expand(dim, -1, -1).size()
        # )
        # 这里仅用简单的DWT将时域信号分为低频和高频两个子带
        x_low = torch.nn.functional.conv1d(
            x, self.dec_lo.expand(dim, -1, -1), stride=self.stride, groups=dim
        )
        x_high = torch.nn.functional.conv1d(
            x, self.dec_hi.expand(dim, -1, -1), stride=self.stride, groups=dim
        )
        # print("stride:",self.stride)
        # print("x:",x.size())
        # print("x_low,x_high:",x_low.size(),x_high.size())
        # TODO
        if x.size()[-1]//self.stride != x_low.size()[-1]:
            x_low = F.interpolate(x_low.to(x.dtype), size=x.size(-1)//self.stride, mode="nearest")
            x_high = F.interpolate(x_high.to(x.dtype), size=x.size(-1)//self.stride, mode="nearest")
        
        # x_low = self.conv_low(x_low)
        # x_high = self.conv_high(x_high)
        
        # print("x_low,x_high:",x_low.size(),x_high.size())
        # print("self.dec_hi.expand(dim, -1, -1):",self.dec_hi.expand(dim, -1, -1).size())
        # x_low_pool = self.low_pooling(x_low)
        # x_high_pool = self.high_pooling(x_high)
        # out = torch.cat([x_low_pool,x_high_pool],dim=-1)
        # print("x_low.size()",x_low.size())
        
        # if self.is_upsample:
        #     # print("size for x_low&x_high before upsample:",x_low.size(),x_high.size())
        #     # upsample the mask using nearest neighbor
        #     x_low = F.interpolate(
        #         x_low.to(x.dtype), size=x.size(-1), mode="nearest"
        #     )
        #     x_high = F.interpolate(
        #         x_high.to(x.dtype), size=x.size(-1), mode="nearest"
        #     )
            # out = 
            # print("size for x_low&x_high after upsample:",x_low.size(),x_high.size())
        
        if self.fusion == 'element_product':
            if self.is_stack:
                out = torch.cat([x_low,x_high],dim=-1)
            else:
                out = x_low * x_high
 
        elif self.fusion == 'element_add':
            out = x_low + x_high
        # elif self.fusion == 'weight_matrix':
        #     pass
        
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out.size(-1), mode="nearest"
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out = out * out_mask.detach()
        out_mask = out_mask.bool()
        return out, out_mask


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the input features
        n_head,  # number of attention heads
        n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # dimension of the hidden layer in MLP
        act_layer=nn.GELU,  #nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for the projection / MLP
        path_pdrop=0.0,  # drop path rate
        mha_win_size=-1,  # > 0 to use window mha
        use_rel_pe=False,  # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        self.ln1_pre = LayerNorm(n_embd)
        self.ln2_pre = LayerNorm(n_embd)

        self.ln1_channel = LayerNorm(n_embd)
        self.ln2_channel = LayerNorm(n_embd)
        
        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.attn_pre = MaskedMHCA(
            n_embd,
            n_head=n_head,
            # n_qx_stride=1,
            # n_kv_stride=1,
            # if u want to ablation whether to need local-attention,use follows:
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
        )
        # self.attn_channel = MaskedMHCA_Channel(
        #     n_embd,
        #     n_head=n_head,
        #     n_qx_stride=1,
        #     n_kv_stride=1,
        #     # n_qx_stride=n_ds_strides[0],
        #     # n_kv_stride=n_ds_strides[1],
        #     attn_pdrop=attn_pdrop,
        #     proj_pdrop=proj_pdrop,
        # )
        # input_pre
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = (
                n_ds_strides[0] + 1,
                n_ds_strides[0],
                (n_ds_strides[0] + 1) // 2,
            )
            self.pool_skip_pre = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding
            )
        else:
            self.pool_skip_pre = nn.Identity()

        # two layer mlp
        # if n_hidden is None:
        #     n_hidden = 4 * n_embd  # default
        # if n_out is None:
        #     n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp_pre = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )
        # self.mlp_channel = nn.Sequential(
        #     nn.Conv1d(n_embd, n_hidden, 1),
        #     act_layer(),
        #     nn.Dropout(proj_pdrop, inplace=True),
        #     nn.Conv1d(n_hidden, n_out, 1),
        #     nn.Dropout(proj_pdrop, inplace=True),
        # )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn_pre = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp_pre = AffineDropPath(n_out, drop_prob=path_pdrop)
            # self.drop_path_attn_channel = AffineDropPath(n_embd, drop_prob=path_pdrop)
            # self.drop_path_mlp_channel = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn_pre = nn.Identity()
            self.drop_path_mlp_pre = nn.Identity()
            # self.drop_path_attn_channel = nn.Identity()
            # self.drop_path_mlp_channel = nn.Identity()
        
    def forward(self, x, mask, pos_embd=None):
        out, out_mask = self.attn_pre(self.ln1_pre(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip_pre(x) * out_mask_float + self.drop_path_attn_pre(out)
        out = out + self.drop_path_mlp_pre(self.mlp(self.ln2_pre(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
    
        return out, out_mask


class TimeTransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,  # dimension of the input features
        n_head,  # number of attention heads
        n_ds_strides=(1, 1),  # downsampling strides for q & x, k & v
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # dimension of the hidden layer in MLP
        act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for the projection / MLP
        path_pdrop=0.0,  # drop path rate
        mha_win_size=-1,  # > 0 to use window mha
        use_rel_pe=False,  # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        self.ln1_pre = LayerNorm(n_embd)
        self.ln2_pre = LayerNorm(n_embd)

        self.ln1_channel = LayerNorm(n_embd)
        self.ln2_channel = LayerNorm(n_embd)

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.attn_pre = TimeMaskedMHCA(
            n_embd,
            n_head=n_head,
            # n_qx_stride=1,
            # n_kv_stride=1,
            # if u want to ablation whether to need local-attention,use follows:
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
        )
        # self.attn_channel = MaskedMHCA_Channel(
        #     n_embd,
        #     n_head=n_head,
        #     n_qx_stride=1,
        #     n_kv_stride=1,
        #     # n_qx_stride=n_ds_strides[0],
        #     # n_kv_stride=n_ds_strides[1],
        #     attn_pdrop=attn_pdrop,
        #     proj_pdrop=proj_pdrop,
        # )
        # input_pre
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = (
                n_ds_strides[0] + 1,
                n_ds_strides[0],
                (n_ds_strides[0] + 1) // 2,
            )
            self.pool_skip_pre = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding
            )
        else:
            self.pool_skip_pre = nn.Identity()
        # self.pool_skip_pre = nn.Identity()
        # self.pool_skip_channel = nn.Identity()

        # two layer mlp
        # if n_hidden is None:
        #     n_hidden = 4 * n_embd  # default
        # if n_out is None:
        #     n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp_pre = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )
        # self.mlp_channel = nn.Sequential(
        #     nn.Conv1d(n_embd, n_hidden, 1),
        #     act_layer(),
        #     nn.Dropout(proj_pdrop, inplace=True),
        #     nn.Conv1d(n_hidden, n_out, 1),
        #     nn.Dropout(proj_pdrop, inplace=True),
        # )

        self.conv_input = nn.Conv1d(n_embd, n_embd, 1)
        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn_pre = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp_pre = AffineDropPath(n_out, drop_prob=path_pdrop)
            # self.drop_path_attn_channel = AffineDropPath(n_embd, drop_prob=path_pdrop)
            # self.drop_path_mlp_channel = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn_pre = nn.Identity()
            self.drop_path_mlp_pre = nn.Identity()
            # self.drop_path_attn_channel = nn.Identity()
            # self.drop_path_mlp_channel = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        out, out_mask = self.attn_pre(self.ln1_pre(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip_pre(x) * out_mask_float + self.drop_path_attn_pre(out)
        # FFN
        out = out + self.drop_path_mlp_pre(self.mlp(self.ln2_pre(out)) * out_mask_float)
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask



class CrossMaskedMHCA(nn.Module):

    def __init__(
        self,
        n_embd,  # dimension of the output features
        n_head,  # number of heads in multi-head self-attention
        n_qx_stride=1,  # dowsampling stride for query and input
        n_kv_stride=1,  # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        print("CrossMaskedMHCA's nhead is: ", self.n_head)
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.query_norm = LayerNorm(self.n_embd)
        # self.query_pooling = nn.MaxPool1d(
        #     kernel_size, stride=stride, padding=padding)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd,
            self.n_embd,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=self.n_embd,
            bias=False,
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, query,key,value,query_mask,kv_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = query.size()
        assert query.size()==key.size()==value.size(),"check"

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(query, query_mask)
        q = self.query_norm(q)
        # qx_mask = qx_mask.transpose(-2, -1)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(key, kv_mask)
        k = self.key_norm(k)
        # kv_mask = kv_mask.transpose(-2, -1)
        v, _ = self.value_conv(value, kv_mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)

        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        # out = self.out_pooling(out)
        return out, qx_mask


class CrossAttention(nn.Module):

    def __init__(
        self,
        n_embd,  # dimension of the input features
        n_head,  # number of attention heads
        n_out=None,  # output dimension, if None, set to input dim
        n_hidden=None,  # dimension of the hidden layer in MLP
        act_layer=nn.GELU,  # nonlinear activation used in MLP, default GELU
        # epic-verb：0.1-0.15-0.25
        # epic-noun：0.0-0.0-0.2
        # thumos14：0.0-0.1-0.13-n_head=2-2/2-4
        # activitynet: 0-0-0  -n_head=2-2/2-4
        attn_pdrop=0.1,  # dropout rate for the attention map  
        proj_pdrop=0.15,  # dropout rate for the projection / MLP  
        path_pdrop=0.25,  # drop path rate  
        use_rel_pe=False,  # if to add rel position encoding to attention
    ):  
        super().__init__()
        self.ln = LayerNorm(n_embd)
        if n_out is None:
            n_out = n_embd

        # print(n_ds_strides)
        self.attn = CrossMaskedMHCA(
            n_embd,
            n_head=n_head,
            n_qx_stride=1,
            n_kv_stride=1,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
        )

        self.pool_skip = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )
    
        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, query,key,value,query_mask,kv_mask, pos_embd=None):
        out, out_mask = self.attn(query,key,value,query_mask,kv_mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(query) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask



# ------------------------------------ tools ------------------------------------ 
class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=3,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        expansion_factor=2,  # expansion factor of feat dims
        n_out=None,  # output dimension, if None, set to input dim
        act_layer=nn.ReLU,  # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

            # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding
        )
        self.conv2 = MaskedConv1D(width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        # print("AffineDropPath")
        return drop_path(self.scale * x, self.drop_prob, self.training)