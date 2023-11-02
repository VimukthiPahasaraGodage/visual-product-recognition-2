import os

import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from transformers import AutoImageProcessor, Dinov2Model


class TensorDataset(Dataset):
    def __init__(self, annotations_file, tensor_dir, transform=None, target_transform=None):
        self.tensor_labels = pd.read_csv(annotations_file)
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.tensor_labels)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, self.tensor_labels.iloc[idx, 1])
        tensor = torch.load(tensor_path)
        label = self.tensor_labels.iloc[idx, 2]
        image_path = self.tensor_labels.iloc[idx, 3]
        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            label = self.target_transform(label)
        return tensor, label, image_path


class DeploymentModel_DINO_v2:
    def __init__(self):
        model_ckpt = "facebook/dinov2-giant"
        self.image_processor = AutoImageProcessor.from_pretrained(model_ckpt,
                                                             do_normalize=True,
                                                             do_center_crop=True,
                                                             do_rescale=True,
                                                             do_resize=True,
                                                             size={'shortest_edge': 384},
                                                             crop_size={'height': 384, 'width': 384})
        model = Dinov2Model.from_pretrained(model_ckpt)
        model = model.to('cuda')
        self.model = model

        self.gallery_tensor_dict = {}
        self.gallery_label_dict = {}
        self.gallery_img_path_dict = {}

        self.load_gallery_embeddings()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DeploymentModel_DINO_v2, cls).__new__(cls)
        return cls.instance

    def img_transform(self, image):
        images = F.resize(image, (384, 384), antialias=True)
        images = self.image_processor(images, return_tensors="pt")
        return images

    def load_gallery_embeddings(self):
        tensor_dataset = TensorDataset("/home/group15/VPR/visual-product-recognition-2/Deployment/Streamlit_Web_App/gallery_tensors_dino_v2/gallery_tensors_dino_v2.csv",
                                       "/home/group15/VPR/visual-product-recognition-2/Deployment/Streamlit_Web_App")
        tensor_dataloader = DataLoader(tensor_dataset, batch_size=1, shuffle=True)

        for idx, data in enumerate(tensor_dataloader):
            tensors, labels, image_paths = data
            self.gallery_tensor_dict[idx] = tensors
            self.gallery_label_dict[idx] = labels
            self.gallery_img_path_dict[idx] = image_paths

    def get_query_embedding(self, image_path, x, y, w, h):
        query_image = read_image(image_path)
        query_image = F.crop(F.to_pil_image(query_image), y, x, h, w)
        preprocessed_query_image = torch.unsqueeze(self.img_transform(query_image), dim=0).to('cuda')
        with torch.no_grad():
            output = self.model(preprocessed_query_image)
            embedding = output.last_hidden_state[:, 0].cpu()
        return embedding

    def get_matching_gallery_images(self, query_tensor):
        tensor_dict = self.gallery_tensor_dict
        img_path_dict = self.gallery_img_path_dict

        query_tensor_ = query_tensor.to('cuda')

        euclidean = None
        for k in range(len(tensor_dict)):
            query_tensor = query_tensor_.repeat(tensor_dict[k].shape[0], 1)
            gallery_tensors = tensor_dict[k].to('cuda')
            if euclidean is None:
                euclidean = torch.nn.functional.pairwise_distance(gallery_tensors, query_tensor, p=2)
            else:
                euclidean = torch.cat(
                    (euclidean, torch.nn.functional.pairwise_distance(gallery_tensors, query_tensor, p=2)), dim=0)

        euclidean_, euclidean_indices = torch.sort(euclidean)
        euclidean_indices = euclidean_indices.cpu().tolist()

        similar_images = []
        for i in range(len(euclidean_indices)):
            similar_images.append(img_path_dict[euclidean_indices[i]][0])

        return similar_images
