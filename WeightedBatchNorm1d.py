"""
Custom version of BatchNorm1d to allow the use of sample weights 
in the computation of batch mean and variance.
"""
import torch


class WeightedBatchNorm1d(torch.nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(WeightedBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x, weights=None):
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            if weights is None:
                mean = x.mean([1])
                var = x.var([1], unbiased=False)
            else:
                var_weights = weights / weights.sum()
                mean = (var_weights * x).sum([0, 1])
                var = (var_weights * (x - mean) ** 2).sum([0, 1])

            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            x = x * self.weight[None, :] + self.bias[None, :]
        return x
