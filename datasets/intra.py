import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from eval_utils.sample_generation import evaluate
import torch
import trimesh
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, *args, split_set="train", **kwargs):
        super().__init__()
        self.mesh_dir = "datasets/intra-m"  # Directory containing vessel mesh files
        self.mesh_files = [f for f in os.listdir(self.mesh_dir) if f.endswith('.ply') or f.endswith('.obj')]  # Load .ply/.obj files
        
        # Split the dataset into train/test
        self.train_files, self.val_files = train_test_split(self.mesh_files, test_size=0.2, random_state=42)
        
        # Set the split based on the argument passed (train or test)
        if split_set == "train":
            self.mesh_files = self.train_files
        elif split_set == "test":
            self.mesh_files = self.val_files
        else:
            raise ValueError("split_set must be 'train' or 'test'")
        
        print(f"Loading {split_set} data with {len(self.mesh_files)} files.")
    
    def __len__(self):
        return len(self.mesh_files)  # Length of either the train or test split

    def __getitem__(self, idx):
        mesh_file = self.mesh_files[idx]  # Load a file from the selected split
        mesh_path = os.path.join(self.mesh_dir, mesh_file)
        
        vertices, faces = self.load_mesh(mesh_path)
        vertices = self.normalize_to_unit_cube(vertices)
        data_dict = {
            'vertices': vertices,
            'faces': faces,
            'shape_idx': torch.tensor(idx, dtype=torch.int64),
        }
        return data_dict

    def load_mesh(self, mesh_path):
        """ Load mesh data from a file (e.g., .ply or .obj). """
        mesh = trimesh.load_mesh(mesh_path)
        vertices = mesh.vertices  # (n_vertices, 3)
        faces = mesh.faces        # (n_faces, 3)
        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.float32)
    
    def normalize_to_unit_cube(self, vertices):
        """
        Normalize mesh vertices to fit in a unit cube [-1, 1]³.
        """
        # Compute the bounding box
        min_coords = vertices.min(axis=0)[0]  # Along each axis
        max_coords = vertices.max(axis=0)[0]
        
        # Compute the centroid and translate the vertices
        centroid = (min_coords + max_coords) / 2
        vertices -= centroid
        
        # Scale the vertices to fit within [-1, 1]³
        max_extent = (max_coords - min_coords).max()
        vertices /= max_extent / 2
        
        return vertices


if __name__ == "__main__":
    dataset = Dataset()
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler = sampler,
        batch_size=1,
        num_workers=4,
        # add for meshgpt
        drop_last = True,
    )
    next(iter(dataloader))