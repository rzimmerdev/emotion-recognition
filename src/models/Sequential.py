from torch import nn


class TransformerClassifier(nn.Module):  # Simple Default Transformer Layer
    def __init__(self, num_classes):
        super().__init__()
        self.classifier_layer = nn.Transformer()

    def forward(self, x):
        pass
