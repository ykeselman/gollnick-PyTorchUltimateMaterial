#%%
from torchvision import transforms
from PIL import Image
# %% import image
fname = '/home/yakov/Studies/gollnick-PyTorchUltimateMaterial/060_CNN_ImageClassification/kiki.jpg'
img = Image.open(fname)
img

# %% compose a series of steps
preprocess_steps = transforms.Compose([
    transforms.Resize(300),  # better (300, 300)
    transforms.RandomRotation(50),
    transforms.CenterCrop(500),
    # not gray-scaled because of errors
    # transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # not normalized for display below
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet values
])
x = preprocess_steps(img)
x

# %% get the mean and std of given image
x.mean([1, 2]), x.std([1, 2])

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(np.transpose(x, (1, 2, 0)))
