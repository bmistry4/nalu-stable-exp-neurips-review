
import torch


def l1(params):
    # calc L1 error over all params in the network
    l1 = 0
    for param in params:
        l1 += torch.norm(param, 1)
    return l1

def huber(params):
    # calc huber score for all params in the network
    huber = 0   # accumulated huber loss
    delta = 1   # manually tuned

    for param in params:
        t = torch.flatten(param)
        mask = torch.abs(t) <= delta
        huber += (t[mask] ** 2 / 2.).sum()
        huber += (delta * (t[~mask].abs() - (0.5*delta))).sum()
    return huber

class Regualizer:
    def __init__(self, support='nac', type='bias', shape='squared', zero=False, zero_epsilon=0):
        super()
        self.zero_epsilon = 0

        if zero:
            self.fn = self._zero
        else:
            identifier = '_'.join(['', support, type, shape])
            self.fn = getattr(self, identifier)

    def __call__(self, W):
        return self.fn(W)

    def _zero(self, W):
        return 0

    def _mnac_bias_linear(self, W):
        return torch.mean(torch.min(
            torch.abs(W - self.zero_epsilon),
            torch.abs(1 - W)
        ))

    def _mnac_bias_squared(self, W):
        return torch.mean((W - self.zero_epsilon)**2 * (1 - W)**2)
    
    def _mnac_bias_none(self, W):
        return self._zero(W)

    def _mnac_oob_linear(self, W):
        return torch.mean(torch.relu(
            torch.abs(W - 0.5 - self.zero_epsilon)
            - 0.5 + self.zero_epsilon
        ))

    def _mnac_oob_squared(self, W):
        return torch.mean(torch.relu(
            torch.abs(W - 0.5 - self.zero_epsilon)
            - 0.5 + self.zero_epsilon
        )**2)

    def _nac_bias_linear(self, W):
        W_abs = torch.abs(W)
        return torch.mean(torch.min(
            W_abs,
            torch.abs(1 - W_abs)
        ))

    def _nac_bias_squared(self, W):
        return torch.mean(W**2 * (1 - torch.abs(W))**2)

    def _nac_bias_none(self, W):
        return self._zero(W)

    def _nac_oob_linear(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1))

    def _nac_oob_squared(self, W):
        return torch.mean(torch.relu(torch.abs(W) - 1)**2)

    def _nac_wset_linear(self, W: list):
        # W refers to W_set 
        w_mean = torch.mean(torch.stack(W), 0)
        reg_err = 0
        for w in W:
            # use square not absolute to match code of https://arxiv.org/abs/1511.08228
            # https://github.com/openai/neural-gpu/blob/master/neuralgpu/model.py#L127
            reg_err += torch.square(w - w_mean).sum()
        # average so result is between 0-1 (different to original paper)
        return reg_err / (len(W) * torch.numel(w))

    def _npu_bias_none(self, W):
        return self._zero(W)

    def _npu_W_linear(self, W: list):
        # W[0] = W_re, W[1] = W_im
        W_re = torch.mean(torch.abs(1 - torch.abs(W[0])))  # {-1,1} not penalised
        if len(W) == 1:
            return W_re
        else:
            return W_re + l1(W[1])  # self._nac_bias_linear(W[1])

    def _realnpu_bias_linear(self, W: list):
        # penalise {-1,0,1} on W_re
        return self._nac_bias_linear(W[0])
