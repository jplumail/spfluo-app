## Segmentation for cleaning

After the [picking part](../picking), we may have overlapping bounding boxes. That's why we developped a 3D segmentation tool that cleans each patch extracted. It is based on a 3D Unet trained on synthetic data.