import torch
import numpy as np
import sys
import pathlib
import scipy
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parent / 'odlstuff'))
from fanbeam import *
from parbeam import *
from functools import partial
import math

CT_NAMES = {'ct_parbeam', 'ct_fanbeam', 'ct_lion_fanbeam', 'ct_lion_parbeam', 'lact'}

class InverseOperator(object):
    def __init__(self, imsize, name, views=10, channels=1, blursize=5, scale_factor=2, ct_impl='astra_cuda'):
        self.imsize = imsize
        self.name = name
        self.views = views
        self.channels = channels
        self.ct_impl = ct_impl
        self.data_gradient_scale = 1.0
        if name == 'ct_parbeam':
            self.radon_sv = parbeam([imsize, imsize], views, 512, impl=ct_impl)
        elif name == 'ct_lion_parbeam':
            angle_step = 2 * np.pi / views
            self.data_gradient_scale = 0.09
            self.radon_sv = parbeam(
                [imsize, imsize],
                views,
                900,
                min_pt=[-150, -150],
                max_pt=[150, 150],
                y_area=[-450, 450],
                angle_min=-0.5 * angle_step,
                angle_max=2 * np.pi - 0.5 * angle_step,
                impl=ct_impl,
            )
        elif name == 'ct_fanbeam':
            self.radon_sv = fanbeam([imsize, imsize], views, 512, impl=ct_impl)
        elif name == 'ct_lion_fanbeam':
            angle_step = 2 * np.pi / views
            self.data_gradient_scale = 0.09
            self.radon_sv = fanbeam(
                [imsize, imsize],
                views,
                900,
                min_pt=[-150, -150],
                max_pt=[150, 150],
                y_area=[-450, 450],
                src_radius=575,
                det_radius=475,
                angle_min=0.5 * angle_step,
                angle_max=2 * np.pi + 0.5 * angle_step,
                src_to_det_init=(1, 0),
                flip_angles=True,
                flip_detector=True,
                impl=ct_impl,
            )
        elif name == 'lact':
            self.radon_sv = parbeam([imsize, imsize], views, 512, lact=True, impl=ct_impl)
        elif name == 'deblur_uniform':
            return NotImplementedError

        elif name == 'super':
            return NotImplementedError

        elif name == 'denoise':
            pass
        else:
            return NotImplementedError

    def A(self, x):
        if self.name in CT_NAMES:
            x2 = torch.unsqueeze(torch.clone(x), 0)
            out = self.radon_sv.Atimes(x2)
            return torch.squeeze(out, dim=0)
        elif self.name == 'denoise':
            return x
        return NotImplementedError

    def AT(self, y):
        if self.name in CT_NAMES:
            y2 = torch.unsqueeze(y, 0)
            out = self.radon_sv.ATtimes(y2)
            return torch.squeeze(out, 0)
        elif self.name == 'denoise':
            return y
        return NotImplementedError

    def Adagger(self, y):
        if self.name in CT_NAMES:
            y2 = torch.unsqueeze(y, 0)
            out = self.radon_sv.Adagger(y2)
            return torch.squeeze(out, 0)
        elif self.name == 'denoise':
            return y
