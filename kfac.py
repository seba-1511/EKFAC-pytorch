import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer


class KFAC(Optimizer):

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0):
        """ K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).

        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.params = []
        self.a_mappings = {}
        self.g_mappings = {}
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                if mod_class == 'Conv2d':
                    if not self.sua:
                        # Adding gathering filter for convolution
                        d['gathering_filter'] = self._get_gathering_filter(mod)
                self.params.append(d)
        super(KFAC, self).__init__(self.params, {})

    def step(self):
        """Performs one step of preconditioning."""
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if self._iteration_counter % self.update_freq == 0:
                self._compute_covs(group, state)
                ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'])
                state['ixxt'] = ixxt
                state['iggt'] = iggt
            else:
                if self.alpha != 1:
                    self._compute_covs(group, state)
            # Preconditionning
            self._precond(weight, bias, group, state)
        self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.a_mappings[mod] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.g_mappings[mod] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            bias.grad.data = gb
            g = g[:, :-1]
        g = g.contiguous().view(*s)
        weight.grad.data = g

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gy = mod.y.grad.data.sum(0, keepdim=True)
            gys = gy.shape
            pool_size = (gys[2] - s[2] // 2 + mod.padding[0],
                         gys[3] - s[3] // 2 + mod.padding[1])
            # Here we compute 1 bias per spatial position of the filter!
            gb = F.avg_pool2d(gy, kernel_size=pool_size, stride=(1, 1),
                              padding=mod.padding, ceil_mode=False,
                              count_include_pad=True).data
            gb *= pool_size[0] * pool_size[1]
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            self._squared_fisher_norm += (bias.grad.data * gb).sum()
            bias.grad.data = gb
            g = g[:, :-1]
        self._squared_fisher_norm += (weight.grad.data * g).sum()
        weight.grad.data = g

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.a_mappings[group['mod']]
        bs = x.shape[0]
        gy = self.g_mappings[group['mod']]
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.conv2d(x, group['gathering_filter'],
                             stride=mod.stride, padding=mod.padding,
                             groups=mod.in_channels)
            x = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(bs)
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(bs))

    def _inv_covs(self, xxt, ggt):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) / float(xxt.shape[0])
            tg = torch.trace(ggt) / float(ggt.shape[0])
            pi = (tx / tg) ** 0.5
        # Regularizes and inverse
        diag_xxt = xxt.new(xxt.shape[0]).fill_(self.eps * pi)
        diag_ggt = ggt.new(ggt.shape[0]).fill_(self.eps / pi)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt

    def _get_gathering_filter(self, mod):
        """Convolution filter that extracts input patches."""
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter
