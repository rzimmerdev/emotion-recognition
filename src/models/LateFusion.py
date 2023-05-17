import torch
from torch import nn, concat, tensor

from .CNN import ResNet, InceptionNet
from .OpticalFlow import dense_optical_flow


class LateMultidimensionalFusion(nn.Module):
    def __init__(self, in_features=1, out_features=8):
        super().__init__()

        self.cnn_raw = InceptionNet(in_features, 32)
        self.cnn_flow = ResNet(in_features, 32)

        self.optical_flow = lambda x: tensor(dense_optical_flow(x)).type(torch.float32)

        self.rnn = nn.LSTM(input_size=64, hidden_size=64, num_layers=2)
        self.out = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=out_features)
        )

    def forward(self, x):

        intervals = torch.split(x[0].type(torch.float32), 10, dim=0)
        predictions = []

        for x in intervals:
            raw_features = self.cnn_raw(x)
            if not isinstance(raw_features, torch.Tensor):
                raw_features = raw_features.logits

            optical_flow = self.optical_flow(x.detach().cpu()).to(device=x.device)
            flow_features = self.cnn_flow(optical_flow)

            features = concat((raw_features, flow_features), dim=1)

            series = self.rnn(features)[0]

            step_prediction = self.out(series)[-1]
            predictions.append(step_prediction)

        return torch.stack(predictions)
