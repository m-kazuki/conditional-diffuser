import diffuser.utils as utils
import ipdb
# from diffuser.datasets.dataset_trajdata import UnifiedDataset
from trajdata import UnifiedDataset
from torch.utils.data import DataLoader
import os
from diffuser.utils.training_ped import Trainer
# from ldm.models.diffusion.ddpm import Diffusion
from ldm.models.diffusion.diffusion import GaussianDiffusion
from ldm.modules.diffusionmodules.trajunet import UNetModel
import time
from collections import defaultdict
import torch
import pickle

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#


# dataset = UnifiedDataset(
#     desired_data=["eupeds_eth", "eupeds_hotel", "eupeds_univ", "eupeds_zara1", "eupeds_zara2"],
#     data_dirs={  # Remember to change this to match your filesystem!
#         "eupeds_eth": "~/traj_dataset/eth-uty",
#         "eupeds_hotel": "~/traj_dataset/eth-uty",
#         "eupeds_univ": "~/traj_dataset/eth-uty",
#         "eupeds_zara1": "~/traj_dataset/eth-uty",
#         "eupeds_zara2": "~/traj_dataset/eth-uty",
#     },
#     history_sec =(0.0,  0.0), 
#     future_sec =(15.9,  15.9),
#     desired_dt=0.1,
#     agent_interaction_distances=defaultdict(lambda: 1e-5),
# )

dataset = UnifiedDataset(
    desired_data=["eupeds_eth", "eupeds_hotel"],
    data_dirs={  # Remember to change this to match your filesystem!
        "eupeds_eth": "~/traj_dataset/eth-uty",
        "eupeds_hotel": "~/traj_dataset/eth-uty"
    },
    history_sec =(0.0,  0.0), 
    future_sec =(15.9,  15.9),
    desired_dt=0.1,
    agent_interaction_distances=defaultdict(lambda: 10),
)


dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=dataset.get_collate_fn(),
    num_workers=1, # This can be set to 0 for single-threaded loading, if desired.
    pin_memory=True
)


#------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

dataset_ego = torch.tensor([])
dataset_neigh = []
print(len(dataloader))

count = 0
for batch in dataloader:
    # print(f'batch load: {timer():8.4f}')
    pos_ego = torch.cat((batch.agent_hist.position, batch.agent_fut.position), axis=1).transpose(1,2)
    vel_ego = torch.cat((batch.agent_hist.velocity, batch.agent_fut.velocity), axis=1).transpose(1,2)
    traj_ego = torch.cat((pos_ego, vel_ego), axis=1)
    dataset_ego = torch.cat((dataset_ego, traj_ego), axis=0)


    futpos = batch.neigh_fut.position
    futvel = batch.neigh_fut.velocity
    if batch.neigh_fut.position.shape[2] != 159:
        fut = batch.neigh_fut.position.shape
        tmp = torch.ones(fut[0], fut[1], 159, fut[3]) * torch.nan
        tmp[:,:,:fut[2],:] = batch.neigh_fut.position
        futpos = tmp

        tmp = torch.ones(fut[0], fut[1], 159, fut[3]) * torch.nan
        tmp[:,:,:fut[2],:] = batch.neigh_fut.velocity
        futvel = tmp
    pos_neigh = torch.cat((batch.neigh_hist.position, futpos), axis=2).transpose(2,3)
    vel_neigh = torch.cat((batch.neigh_hist.velocity, futvel), axis=2).transpose(2,3)
    traj_neigh = torch.cat((pos_neigh, vel_neigh), axis=2)
    dataset_neigh.append(traj_neigh)

    if count % 1000 == 0:
        print(count)

    count+=1

f = open('dataset_ego.txt', 'wb')
pickle.dump(dataset_ego.tolist(), f)
f.close()
f = open('dataset_neigh.txt', 'wb')
pickle.dump(dataset_neigh, f)
f.close()