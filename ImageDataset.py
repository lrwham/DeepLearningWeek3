import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ImageDataset(Dataset):
    def __init__(self, labels, image_folder, transform=None):
        self.labels = labels
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get image path and label
        img_name = os.path.join(self.image_folder, self.labels.iloc[idx, 0])
        label = self.labels.iloc[idx, 1]

        # Open and transform image
        image = Image.open(img_name + ".tif").convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)
        return image, label
