import diffuser.utils as utils
import ipdb
# from diffuser.datasets.dataset_trajdata import UnifiedDataset
# from trajdata import UnifiedDataset
from torch.utils.data import DataLoader
import os
from diffuser.utils.training_ped import Trainer
# from ldm.models.diffusion.ddpm import Diffusion
from ldm.models.diffusion.diffusion import GaussianDiffusion
from ldm.modules.diffusionmodules.trajunet import UNetModel
import time
from collections import defaultdict
import torch


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#


class LightDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

        self.n_data = self.data.shape[0]
    
    def __len__(self):
        return self.n_data
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = LightDataset("./ethuty.pt")

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=1, # This can be set to 0 for single-threaded loading, if desired.
    pin_memory=True
)



#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

horizon = 160

# diffusion = Diffusion(
#     horizon = horizon
# ).cuda()

unet = UNetModel(
    horizon = horizon
).cuda()

diffusion = GaussianDiffusion(
    model = unet,
    horizon = horizon
).cuda()

timestr = time.strftime("%Y%m%d-%H%M")
output_path = "./logs/eupeds/"+timestr
os.makedirs(output_path, exist_ok=True)

trainer = Trainer(
    diffusion_model=diffusion,
    data_loader=dataloader,
    results_folder=output_path
)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

# print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(dataset[0])
# loss, _ = diffusion.loss(*batch)
# loss.backward()
# print('✓')

#-----------------------------------------------------------------------------#
#----------------------------- training settings -----------------------------#
#-----------------------------------------------------------------------------#

n_epochs = 100

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs}')
    trainer.train(i)

