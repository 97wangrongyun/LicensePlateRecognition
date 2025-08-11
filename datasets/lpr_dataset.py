import os
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms


class CCPDLPRDataset(Dataset):
    """
    CCPD2019 数据集的 PyTorch Dataset 类
    假设图像文件名中包含车牌号，如：XXXXX_粤B12345.jpg
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) 
                            for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def _extract_plate_from_filename(self, filename):
        """
        从文件名中提取车牌字符串，例如：粤B12345
        """
        try:
            name = os.path.splitext(os.path.basename(filename))[0]
            plate = name.split('_')[-1]  # 默认车牌号是最后一段
            return plate
        except Exception as e:
            print(f\"[WARN] Failed to parse plate from filename: {filename}\")
            return \"UNKNOWN\"

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self._extract_plate_from_filename(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label
