import torch
import torch.nn.functional as func

from .misc import LayerNorm2d
from .utils import replace_layers


class SimpleGate(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(torch.torch.nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        # Simplified Channel Attention
        self.sca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1, padding=0,
                stride=1,
                groups=1,
                bias=True
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = torch.nn.Dropout(drop_out_rate) if drop_out_rate > 0. else torch.nn.Identity()
        self.dropout2 = torch.nn.Dropout(drop_out_rate) if drop_out_rate > 0. else torch.nn.Identity()

        self.beta = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(torch.nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        if enc_blk_nums is None:
            enc_blk_nums = []

        if dec_blk_nums is None:
            dec_blk_nums = []

        super().__init__()

        self.intro = torch.nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )
        self.ending = torch.nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.middle_blks = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                torch.nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                torch.nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            torch.nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(chan, chan * 2, 1, bias=False),
                    torch.nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                torch.nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor):
        # noinspection PyPep8Naming
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = func.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class LocalBase(object):
    def convert(self, *args, train_size, **kwargs):
        # noinspection PyTypeChecker
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            # noinspection PyUnresolvedReferences
            self.forward(imgs)


class NAFNetLocal(LocalBase, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        LocalBase.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        # noinspection PyPep8Naming
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
