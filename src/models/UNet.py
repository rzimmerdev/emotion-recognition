from torch import nn, flatten


class CNNColor(nn.Module):  # UNET
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.feature_layers = [input_channels, 32, 64]
        self.kernels = [5, 5]
        self.pools = [2, 3]

        self.feature_activations = [nn.ReLU for _ in range(len(self.feature_layers) - 1)]

        self.classifier_layers = [(48 // (2 * 3)) ** 2 * self.feature_layers[-1],
                                  120, 84, num_classes]
        self.classifier_activations = [nn.ReLU for _ in range(len(self.classifier_layers) - 1)]
        self.out = nn.Softmax(dim=0)

        feature_layers = []
        for idx, layer in enumerate(list(zip(self.feature_layers[:-1], self.feature_layers[1:]))):
            feature_layers.append(
                nn.Conv2d(in_channels=layer[0], out_channels=layer[1],
                          kernel_size=self.kernels[idx], padding=self.kernels[idx] // 2)
            )
            feature_layers.append(self.feature_activations[idx]())
            feature_layers.append(nn.MaxPool2d(kernel_size=self.pools[idx]))

        classifier_layers = []
        for idx, layer in enumerate(list(zip(self.classifier_layers[:-1], self.classifier_layers[1:]))):
            classifier_layers.append(
                nn.Linear(in_features=layer[0], out_features=layer[1])
            )
            if idx < len(self.classifier_activations) - 1:
                classifier_layers.append(self.classifier_activations[idx]())

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        y = self.classifier(flatten(x))
        return self.out(y)
