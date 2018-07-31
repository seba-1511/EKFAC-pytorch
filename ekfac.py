import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class EKFAC(Optimizer):

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=0.75, diag='ra'):
        """ EKFAC Preconditionner

        Computes the EKFAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Apply SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): running average coefficient
            diag (str): compute the diagonal stats using either a running
                        average ('ra') or the intra minibatch stats ('intra')
        """
        self.a_mappings = dict()
        self.g_mappings = dict()
        self.eps = eps
        self.sua = sua  # the code always perform SUA for now
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.diag_statistic = diag

        self.params = []
        self._iteration_counter = 0

        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                if mod.bias is not None:
                    self.params.append(dict(params=[mod.bias, mod.weight],
                                            mod=mod, layer_type=mod_class))
                else:
                    self.params.append(dict(params=[mod.weight],
                                            mod=mod, layer_type=mod_class))
        super(EKFAC, self).__init__(self.params, {})

    def _save_input(self, mod, i):
        self.a_mappings[mod] = i[0].data

    def _save_grad_output(self, mod, grad_input, grad_output):
        # Note that we store the average gradient, not the sum here
        self.g_mappings[mod] = grad_output[0].data

    def step(self, update_stats=True, update_params=False):
        for group in self.param_groups:
            if group['layer_type'] in ['Conv2d', 'Linear']:
                if len(group['params']) == 2:
                    bias, weight = group['params']
                else:
                    weight = group['params'][0]
                    bias = None
                state = self.state[weight]
                if (update_stats and
                        self._iteration_counter % self.update_freq == 0):
                    # Update Stats
                    if group['layer_type'] == 'Conv2d':
                        self._compute_covs_conv2d(group, state)
                    elif group['layer_type'] == 'Linear':
                        self._compute_covs_linear(group, state)
                    # Inverses
                    ixxt, iggt, ex, vx, eg, vg = self._inv_covs(state['xxt'],
                                                                state['ggt'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                    state['vx'] = vx
                    state['ex'] = ex
                    state['vg'] = vg
                    state['eg'] = eg

                    if group['layer_type'] == 'Conv2d':
                        varD = torch.ger(eg, ex).unsqueeze_(2).unsqueeze_(3)
                        state['varD'] = varD.expand_as(weight)
                        state['ED'] = torch.zeros_like(state['varD'])
                        eps = torch.ger(eg + self.eps, ex + self.eps)
                        eps = eps.unsqueeze_(2).unsqueeze_(3)
                        state['eps'] = eps.expand_as(weight) - state['varD']
                    elif group['layer_type'] in ['Linear']:
                        state['varD'] = torch.ger(eg, ex)
                        eps = torch.ger(eg + self.eps, ex + self.eps)
                        state['ED'] = torch.zeros_like(state['varD'])
                        state['eps'] = eps - state['varD']

                # Preconditionning
                if update_params:
                    if group['layer_type'] == 'Conv2d':
                        self._precond_conv2d(weight, bias, state['ixxt'],
                                             state['iggt'], group['mod'],
                                             state)
                    elif group['layer_type'] == 'Linear':
                        self._precond_linear(weight, bias, state['ixxt'],
                                             state['iggt'], group['mod'],
                                             state)
        self._iteration_counter += 1

    def _precond_linear(self, weight, bias, ixxt, iggt, mod, state):
        gw = weight.grad.data
        gb = bias.grad.data
        bs = self.a_mappings[mod].shape[0]
        varD = state['varD']
        ED = state['ED']
        v_a = state['vx']
        v_g = state['vg']

        g = gw
        if bias is not None:
            g = torch.cat([gw, gb.unsqueeze(1)], dim=1)
        d = torch.mm(torch.mm(v_g.t(), g), v_a)  # to diag space

        if self.diag_statistic == 'ra':
            ED.mul_(self.alpha).add_(1. - self.alpha, d)  # Update RA
            varD.mul_(self.alpha).add_((1. - self.alpha) * bs, (d - ED)**2)
        else:
            a_stats = self.a_mappings[mod]  # Compute M2
            g_stats = self.g_mappings[mod] * bs
            if bias is None:
                a_h = a_stats
            else:
                ones = a_stats.new_ones((a_stats.size(0), 1))
                a_h = torch.cat([a_stats, ones], dim=1)
            ah_f = torch.mm(a_h, v_a)
            g_f = torch.mm(g_stats, v_g)
            d_m2 = torch.mm((g_f**2).transpose(1, 0), ah_f**2) / bs

            ED.mul_(0)  # Update RA
            varD.mul_(self.alpha).add_((1. - self.alpha), d_m2)

        d /= varD + ED**2 + state['eps']  # inverse in diag space
        g = torch.mm(torch.mm(v_g, d), v_a.t())  # back to origin space

        if bias is not None:
            dw = g[:, :-1]
            db = g[:, -1]
            bias.grad.data = db
            weight.grad.data = dw
        else:
            weight.grad.data = g

    def _precond_conv2d(self, weight, bias, ixxt, iggt, mod, state):
        g = weight.grad.data
        s = g.shape
        bs = self.a_mappings[mod].shape[0]
        varD = state['varD']
        ED = state['ED']
        v_a = state['vx']
        v_g = state['vg']

        # KFAC SUA
        g = g.permute(1, 0, 2, 3)
        if bias is not None:
            gy = self.g_mappings[mod].data.sum(0, keepdim=True)
            gys = gy.shape
            pool_size = (gys[2] - s[2] // 2 + mod.padding[0],
                         gys[3] - s[3] // 2 + mod.padding[1])
            # Here we compute 1 bias per spatial position of the filter!
            gb = F.avg_pool2d(gy, kernel_size=pool_size,
                              stride=(1, 1), padding=mod.padding,
                              ceil_mode=False, count_include_pad=True)
            gb *= pool_size[0] * pool_size[1]
            g = torch.cat([g, gb], dim=0)

        g = g.permute(1, 0, 2, 3)
        d = to_space_c_c(g.contiguous(), v_g, v_a)  # to diag space

        if self.diag_statistic == 'ra':
            ED.mul_(self.alpha).add_(1. - self.alpha, d)  # Update RAe
            varD.mul_(self.alpha).add_((1. - self.alpha) * bs, (d - ED)**2)
        else:
            g2 = (self.g_mappings[mod] * bs)  # Compute M2
            a = self.a_mappings[mod]
            g_s = g2.size()

            if bias is not None:
                ones = torch.ones_like(a[:, :1, :, :])
                a_h = torch.cat([a, ones], dim=1)
            else:
                a_h = a
            a_s = a_h.size()
            a_h_p = a_h.permute(0, 2, 3, 1).contiguous().view(-1, a_s[1])
            a_h_p_f = torch.mm(a_h_p, v_a)  # to function space
            a_h_f = a_h_p_f.view(a_s[0], a_s[2], a_s[3], -1)
            a_h_f = a_h_f.permute(0, 3, 1, 2).contiguous()

            g_p = g2.permute(0, 2, 3, 1).contiguous().view(-1, g_s[1])
            g_p_f = torch.mm(g_p, v_g)  # to function space
            g_f = g_p_f.view(g_s[0], g_s[2], g_s[3], -1)
            g_f = g_f.permute(0, 3, 1, 2).contiguous()

            d_m2 = grad_wrt_kernel(a_h_f**2, (g_f**2), mod.padding, mod.stride)
            d_m2 /= bs

            ED.mul_(0)  # Update RA
            varD.mul_(self.alpha).add_((1. - self.alpha), d_m2)

        d /= (varD + ED**2 + state['eps'])  # inverse in diag space
        g = to_space_c_c(d.contiguous(), v_g.t(), v_a.t())  # back to origin

        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            bias.grad.data = gb
            g = g[:, :-1]
        weight.grad.data = g

    def _compute_covs_linear(self, group, state):
        gy = self.g_mappings[group['mod']]
        x = self.a_mappings[group['mod']]
        bs = gy.size(0)
        x_h = x
        if group['mod'].bias is not None:
            ones = x.new_ones((x.size(0), 1))
            x_h = torch.cat([x, ones], dim=1)
        gyr = gy * bs
        state['xxt'] = torch.mm(x_h.t(), x_h) / float(x_h.shape[0])
        state['ggt'] = torch.mm(gyr.t(), gyr) / bs

    def _compute_covs_conv2d(self, group, state):
        mod = group['mod']
        x = self.a_mappings[group['mod']]
        gy = self.g_mappings[group['mod']]
        bs = x.shape[0]
        xr = x.data.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
        if mod.bias is not None:
            ones = torch.ones_like(xr[:1])
            xr = torch.cat([xr, ones], dim=0)
        state['xxt'] = torch.mm(xr, xr.t()) / float(xr.shape[1])
        # Computation of ggt (same for classical KFAC and KFAC SUA)
        gyr = gy.data.permute(1, 0, 2, 3)
        gyr = gyr.contiguous().view(gy.shape[1], -1) * bs
        state['ggt'] = torch.mm(gyr, gyr.t()) / bs

    def _inv_covs(self, xxt, ggt):
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) / float(xxt.shape[0])
            tg = torch.trace(ggt) / float(ggt.shape[0])
            pi = (tx / tg) ** 0.5
        _, ex, vx = torch.svd(xxt, True)
        dx = torch.diag(1. / (ex + self.eps * pi))
        ixxt = torch.mm(vx, torch.mm(dx, vx.t()))
        _, eg, vg = torch.svd(ggt, True)
        dg = torch.diag(1. / (eg + self.eps / pi))
        iggt = torch.mm(vg, torch.mm(dg, vg.t()))
        return ixxt, iggt, ex, vx, eg, vg

    def _get_gathering_filter(self, mod):
        kw, kh = mod.kernel_size
        g_filter = mod.weight.data.new(kw * kh * mod.in_channels, 1, kw, kh)
        g_filter.fill_(0)
        for i in range(mod.in_channels):
            for j in range(kw):
                for k in range(kh):
                    g_filter[k + kh*j + kw*kh*i, 0, j, k] = 1
        return g_filter


def to_space_c_c(x, va, vb):
    # change space version channel/channel
    sx = x.size()
    x = torch.mm(va.t(), x.view(sx[0], -1))
    x = x.view(va.shape[1], sx[1], sx[2], sx[3])
    x = torch.mm(x.permute(0, 2, 3, 1).contiguous().view(-1, sx[1]), vb)
    x = x.view(va.shape[1], sx[2], sx[3], vb.shape[1]).permute(0, 3, 1, 2)
    return x


def grad_wrt_kernel(a, g, padding, stride, target_size=None):
    gk = F.conv2d(a.transpose(0, 1), g.transpose(0, 1).contiguous(),
                  padding=padding, dilation=stride).transpose(0, 1)
    if target_size is not None and target_size != gk.size():
        return gk[:, :, :target_size[2], :target_size[3]].contiguous()
    return gk
