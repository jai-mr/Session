import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, numb_patch, fn):
        super().__init__()
        self.norm = nn.LayerNorm([dim, numb_patch, numb_patch])
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=hidden_dim, kernel_size=(1, 1), padding=0, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=hidden_dim, out_channels=dim, kernel_size=(1, 1), padding=0, bias=False),
            nn.Dropout(dropout)
        )        
    def forward(self, x):
        out = self.net(x)
        return out
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.head_channels = dim
        in_channels = dim
        out_channels = dim
        self.attend = nn.Softmax(dim = -1)
        self.to_keys = nn.Conv2d(in_channels, out_channels, 1)
        self.to_queries = nn.Conv2d(in_channels, out_channels, 1)
        self.to_values = nn.Conv2d(in_channels, out_channels, 1)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b = x.shape[0]
        k = self.to_keys(x).view(b, self.heads, self.head_channels, -1)
        q = self.to_queries(x).view(b, self.heads, self.head_channels, -1)
        v = self.to_values(x).view(b, self.heads, self.head_channels, -1)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3)
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, numb_patch, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, numb_patch, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, numb_patch, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, numb_patch, 
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patch = (image_height // patch_height)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Conv2d(in_channels=channels,
                                            out_channels=patch_dim,
                                            kernel_size=patch_size,
                                            stride=patch_size,
                                            padding=0)

        self.pos_embedding = nn.Parameter(torch.randn(1, patch_dim + 1, self.num_patch, self.num_patch))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, numb_patch, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm([num_patches, 1, 1]),
            nn.Conv2d(in_channels=num_patches, out_channels=num_classes, kernel_size=(1, 1), padding=0, bias=False)
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, _, _, _ = x.shape

        cls_tokens = nn.Parameter(torch.ones(b, 1, self.num_patch, self.num_patch),requires_grad=True)
        cls_tokens = cls_tokens.to(device='cuda')
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding

        x = self.dropout(x)

        x = self.transformer(x)
        x = self.flatten(x)
        x = x[:, 0]
        x = self.to_latent(x)
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        out = self.mlp_head(x)
        out = out.view(-1, 10)
        return out
        