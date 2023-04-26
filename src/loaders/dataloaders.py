from torch.utils.data import DataLoader, random_split


class ImageDataloader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=True, num_workers=8):
        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers)


class VideoDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size, shuffle, num_workers=num_workers)


def get_dataloaders(dataset, dataloader=ImageDataloader, test_data=None):
    sizes = [0.7, 0.2, 0.1]
    train_data, validate_data = random_split(dataset, sizes)

    # For 8 CPU cores
    return dataloader(train_data), dataloader(validate_data), dataloader(test_data) if test_data else None
