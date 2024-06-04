import torch 
from torch.nn.functional import conv1d

import torchaudio
from torchaudio import transforms as audtr

class ExpandSparsifyRepresentation:
    def __init__(self, in_channels, out_channels, time_kernel, \
                stride=1, padding=0, sparsity=100, seed=0):
        """ 
        in_channels: int specifying number of input features
        out_channels: int specifying number of output features
        time_kernel: int specifying uniform kernel over window of time_kernel
                    or, a tensor of shape (window_length,)
        sparsity: int specifying sparsity of representation
        """
        torch.manual_seed(seed)
        if type(time_kernel) is int:
            time_kernel = torch.ones(time_kernel, dtype=torch.float)
        window_length = len(time_kernel)

        gaussians = torch.randn(window_length, out_channels, in_channels)
        ortho_projs = torch.svd(gaussians)[0]
        self.filters = torch.einsum('woi,w->oiw', ortho_projs, time_kernel)
        self.stride = stride 
        self.padding = padding
        self.sparsity = sparsity

    def expand(self, specs):
        """
        specs: tensor of shape (n, in_channels, duration)
        """
        return conv1d(specs, self.filters, stride=self.stride, \
                    padding=self.padding)

    def sparsify(self, representation):
        """ 
        representation: tensor of shape (n, out_channels, duration')
        output: tensor of shape (n, out_channels, duration') where 
                axis=1 (out_channels) has sparsity level specified in __init__
        """
        shape = representation.shape
        topk = torch.topk(representation, self.sparsity, dim=1).values
        topk = torch.min(topk, dim=1).values.reshape(-1)
        topk = torch.einsum('nod,d->nod', torch.ones(shape), topk)
        return representation * (representation >= topk)
    
    def __call__(self, sequence):
        return self.sparsify(self.expand(sequence))


class Representer:
    """
    Usage: given `audio`, a torch.Tensor of shape (1, 194, duration)
    > representer = Representer(n_mels=194, sample_rate=32_000, n_fft=1400)
    > mel = representer(audio)
    """
    def __init__(self, n_mels, sample_rate, n_fft):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft

        self.convert_to_mel = audtr.MelSpectrogram(sample_rate=sample_rate,\
                                                n_mels=n_mels, n_fft=n_fft)
        self.decibel_convert = audtr.AmplitudeToDB(stype="power")

        self.expand_sparsify = ExpandSparsifyRepresentation(n_mels, 2000, 3)

    def __call__(self, waveform):
        mel = self.decibel_convert(self.convert_to_mel(waveform))
        mean, std = mel.mean(), mel.std()
        whitened = torch.sigmoid((mel - mean) / (std + 1e-6))
        return self.expand_sparsify(whitened)
