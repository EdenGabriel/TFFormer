import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock,TimeTransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch=(2, 2, 5),        # (#convs, #stem transformers, #branch transformers)
        mha_win_size=[-1]*6,   # size of local window for mha
        scale_factor=2,      # dowsampling rate for the branch
        with_ln=False,       # if to attach layernorm after conv
        attn_pdrop=0.0,      # dropout rate for the attention map
        proj_pdrop=0.0,      # dropout rate for the projection / MLP
        path_pdrop=0.0,      # droput rate for drop path
        use_abs_pe=False,    # use absolute position embedding
        use_rel_pe=False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(
                n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(
                self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        stem_attn_heads=2
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            #FIXME
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head=stem_attn_heads,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )
        print("--------> stem block done...")
        # FIXME 2-2/2 
        # main branch using transformer with pooling
        frqe_or_time_branch_attn_heads=2

        self.freq_branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.freq_branch.append(
                TransformerBlock(
                    n_embd, n_head=frqe_or_time_branch_attn_heads,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        self.time_branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.time_branch.append(
                TimeTransformerBlock(
                    n_embd, n_head=frqe_or_time_branch_attn_heads,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)




    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        # print("***********************:",x.size())
        B, C, T = x.size()
        # print("backbone----->x:",x.size())
        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        
        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
        # print("convolutional embedding:",x.shape,x[-1].shape)
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # 这里得到的是embedding之后的
        freq_start_x,freq_start_mask = x,mask
        # freq_start_x,freq_start_mask  = self.stack_freq(freq_start_x,freq_start_mask)
        
        # print("\033[1;96m----- stem transformer -----")
        # stem transformer
        for idx in range(len(self.stem)):
            # x, mask, _ = self.stem[idx](x, mask)
            x, mask = self.stem[idx](x, mask)
            # print("stem_transformer[%d] output:" % idx, x.size())
        # print("stem transformer:",x.shape,x[-1].shape)
        
        # prep for outputs
        time_start_x,time_start_mask = x,mask
        

        # freq branch 从convEmbed之后开始
        freq_out_feats = (freq_start_x, )
        freq_out_masks = (freq_start_mask, )
        # tmp_feats = ()
        # print("----- main branch -----")
        # main branch with downsampling
        for idx in range(len(self.freq_branch)):
            # x, mask = self.freq_branch[idx](x, mask)
            freq_start_x, freq_start_mask = self.freq_branch[idx](freq_start_x, freq_start_mask)
            # print("frequency branch[%d] output:" % idx, freq_start_x.size())
            freq_out_feats += (freq_start_x, )
            freq_out_masks += (freq_start_mask, )
            # tmp_feats += (tmpfeats,)


        # print("start_x:",start_x.size())
        # FIXME: 由于python的动态特性，这里注意先把2304对应的向量保存下来
        import time
        start_time = time.time()

        time_out_feats = (time_start_x,)
        time_out_masks = (time_start_mask,)
        for idx in range(len(self.time_branch)):
            time_start_x, time_start_mask = self.time_branch[idx](time_start_x, time_start_mask)
            # print("time branch[%d] output:" % idx, time_start_x.size())
            time_out_feats += (time_start_x, )
            time_out_masks += (time_start_mask, )
        end_time = time.time()
        '''
        print("程序运行时间：", end_time - start_time, "秒")
        # 指定要写入的JSON文件路径
        import json
        file_path = "run_time.json"

        
        # 将数据写入JSON文件
        with open(file_path, "a") as file:
            json.dump(end_time - start_time, file)
            file.write("\n") 
        '''
        # print("\033[0m")
        # res_out_feats = ()
        # for idx, feats in enumerate(tmp_feats):
        #     res_out_feats += (out_feats[idx]+feats,)
        # res_out_feats += (out_feats[-1],)
        # print(res_out_feats[0] == out_feats[0]+tmp_feats[0])
        # return out_feats, out_masks
        
        return freq_out_feats, freq_out_masks, time_out_feats,time_out_masks
        # return freq_out_feats,freq_out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """

    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch=(2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor=2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(
                n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks
