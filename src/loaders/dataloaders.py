from typing import Optional

from torch.utils.data import DataLoader, random_split


class ImageDataloader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=True):
        super().__init__(dataset, batch_size, shuffle)


def get_dataloaders(dataset, test_data=None):
    sizes = [0.7, 0.2, 0.1]
    train_data, validate_data = random_split(dataset, sizes)

    # For 8 CPU cores
    return DataLoader(train_data, num_workers=8), \
        DataLoader(validate_data, num_workers=8), \
        DataLoader(test_data, num_workers=8) if test_data else None
