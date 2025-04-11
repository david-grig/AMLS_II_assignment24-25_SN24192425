import os
import numpy as np
from torch.utils.data import Dataset


class ContrailsDataset(Dataset):
    def __init__(self, root_dir, selected_bands=[11, 14, 15, 16], time_step=4, transform=None):
        self.observation_folders = [os.path.join(root_dir, f)
                                    for f in os.listdir(root_dir)
                                    if os.path.isdir(os.path.join(root_dir, f))]
        self.selected_bands = selected_bands
        self.time_step = time_step
        self.transform = transform

    def __len__(self):
        return len(self.observation_folders)

    def __getitem__(self, idx):
        obs_folder = self.observation_folders[idx]

        # Load selected bands
        band_images = []
        for band_num in self.selected_bands:
            band_path = os.path.join(obs_folder, f"band_{band_num:02d}.npy")
            band_data = np.load(band_path)  # Shape: (Height, Width, Time)
            band_images.append(band_data[..., self.time_step])  # Select time step

        # Stack bands along channel dimension
        image = np.stack(band_images, axis=-1)

        # Load mask
        mask_path = os.path.join(obs_folder, "human_pixel_masks.npy")
        mask = np.load(mask_path).astype(np.float32)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image.float(), mask.float()