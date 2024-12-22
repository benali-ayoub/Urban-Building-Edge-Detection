import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_pipelines):
        super(DQN, self).__init__()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((64, 64))

        # Feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv_out_size = self._get_conv_out((64, 64))

        # Pipeline selection head
        self.pipeline_head = nn.Sequential(
            nn.Linear(self.conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_pipelines)
        )

        # Parameter prediction head
        self.param_head = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, 1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.adaptive_pool(x)
        conv_out = self.conv(x).view(x.size()[0], -1)
        pipeline_logits = self.pipeline_head(conv_out)
        params = self.param_head(conv_out)
        return pipeline_logits, params