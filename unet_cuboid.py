# unet_cuboid.py
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
import math

# (SinusoidalPositionEmbeddings, CuboidAttention, Residual, etc. remain the same)
# ... paste all the helper classes from the previous version of unet_cuboid.py here ...
def exists(x):
    return x is not None

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CuboidAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, cuboid_size=(4, 8, 8), global_tokens=8):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cuboid_size = cuboid_size
        self.global_tokens = global_tokens
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.global_context_tokens = nn.Parameter(torch.randn(1, global_tokens, dim))
    def forward(self, x):
        b, t, h, w, d = x.shape
        ct, ch, cw = self.cuboid_size
        pad_t = (ct - t % ct) % ct
        pad_h = (ch - h % ch) % ch
        pad_w = (cw - w % cw) % cw
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')
        _, nt, nh, nw, _ = x.shape
        x_cuboids = rearrange(x, 'b (t ct) (h ch) (w cw) d -> (b t h w) (ct ch cw) d', ct=ct, ch=ch, cw=cw)
        global_tokens = repeat(self.global_context_tokens, '1 n d -> b n d', b=x_cuboids.shape[0])
        x_cuboids = torch.cat((global_tokens, x_cuboids), dim=1)
        qkv = self.to_qkv(x_cuboids).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out[:, self.global_tokens:, :]
        out = rearrange(out, '(b t h w) (ct ch cw) d -> b (t ct) (h ch) (w cw) d', t=nt//ct, h=nh//ch, w=nw//cw, ct=ct, ch=ch, cw=cw)
        return out[:, :t, :h, :w, :]

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
    
class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None)
        self.ds_conv = nn.Conv3d(dim, dim, (1, 7, 7), padding=(0, 3, 3), groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, (1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(), nn.GroupNorm(1, dim_out * mult),
            nn.Conv3d(dim_out * mult, dim_out, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


# The main U-Net Model
class Unet3D(nn.Module):
    def __init__(self, dim, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1, with_time_emb=True, out_frames=None):
        super().__init__()
        self.channels = channels
        self.out_frames = out_frames if exists(out_frames) else None

        init_dim = init_dim if exists(init_dim) else dim
        self.init_conv = nn.Conv3d(channels, init_dim, (1, 7, 7), padding=(0, 3, 3))
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(lambda x: rearrange(CuboidAttention(dim_out)(rearrange(x, 'b c t h w -> b t h w c')), 'b t h w c -> b c t h w')),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(lambda x: rearrange(CuboidAttention(mid_dim)(rearrange(x, 'b c t h w -> b t h w c')), 'b t h w c -> b c t h w'))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(lambda x: rearrange(CuboidAttention(dim_in)(rearrange(x, 'b c t h w -> b t h w c')), 'b t h w c -> b c t h w')),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = out_dim if exists(out_dim) else channels
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim), nn.Conv3d(dim, out_dim, 1)
        )
        
        # New adapter to control output frames
        if exists(self.out_frames):
            self.final_time_adapter = nn.Conv3d(dim_mults[-1], self.out_frames, kernel_size=1)


    def forward(self, x, time=None):
        in_frames = x.shape[2]
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # New logic to adapt time dimension if needed
        if exists(self.out_frames) and in_frames != self.out_frames:
             x = rearrange(x, 'b c t h w -> b (c t) h w')
             x = self.final_time_adapter(x)
             x = rearrange(x, 'b c h w -> b c 1 h w') # A simple adapter, more complex methods exist
             # This part may need more sophistication depending on the task
             # For now, we assume this reshapes the temporal context
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)