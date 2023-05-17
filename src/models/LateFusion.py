import torch
from torch import nn, concat, tensor

from .CNN import ResNet, InceptionNet
from .OpticalFlow import dense_optical_flow


class LateMultidimensionalFusion(nn.Module):
    def __init__(self, in_features=1, out_features=8, device="cpu"):
        super().__init__()

        self.cnn_raw = InceptionNet(in_features, 32).to(device)
        self.cnn_flow = ResNet(in_features, 32).to(device)

        self.optical_flow = lambda x: tensor(dense_optical_flow(x.detach().cpu().to(torch.float32))) \
            .to(device=x.device, dtype=x.dtype)

        self.rnn = nn.LSTM(input_size=64, hidden_size=64, num_layers=2).to(device)
        self.out = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=out_features)
        ).to(device)

    def forward(self, x):
        intervals = torch.split(x[0].type(x.dtype), 10, dim=0)
        predictions = []

        for x in intervals:
            raw_features = self.cnn_raw(x)
            if not isinstance(raw_features, torch.Tensor) and not isinstance(raw_features, torch.HalfTensor):
                raw_features = raw_features.logits

            optical_flow = self.optical_flow(x)
            flow_features = self.cnn_flow(optical_flow)

            features = concat((raw_features, flow_features), dim=1)

            series = self.rnn(features)[0]

            step_prediction = self.out(series)[-1]
            predictions.append(step_prediction)

        return torch.stack(predictions)
