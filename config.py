import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Data Preprocessing Configs
# Chains together a list of transformations to be applied to each image in sequence.
transform = transforms.Compose([
    # The first convolutional layer filters the 224 × 224 × 3 input image with 96 kernels of size 11 × 11 × 3 with a stride of 4 pixels
    # Resize inputs to 224 × 224
    transforms.Resize((224, 224)),
    # Converts images to tensors -  Converts images from a range of [0,255][0,255] integers to a range of [0,1][0,1] floats and changes them from PIL images to PyTorch tensors.
    transforms.ToTensor(),
    # Normalize for ImageNet models -  Normalizes the tensor to have a mean and standard deviation corresponding to ImageNet, with each color channel normalized independently.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Training Configs
epochs = 100
# Specifies the batch size of 64 images, meaning each iteration of training will process 64 images at once.
batch_size = 64
learning_rate = 0.001
imagenet_data_dir_train = "./ImageNet-Mini/train"
imagenet_data_dir_test = "./ImageNet-Mini/test"
validation_split = 0.2
shuffle = True  # DataLoader.shuffle
# Sets the number of CPU cores for loading data, speeding up data loading by using multiple parallel workers.
num_workers = 4  # DataLoader.num_workers
# Indicates that there are 1,000 classes in MiniImageNet, matching the number of output classes expected for the classification task.
n_classes = 1000


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')