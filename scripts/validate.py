import os
from torch.utils.data import DataLoader
import torch
import ipdb

from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset

import sys

# sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '.'))

from  trajdata_utils.interactive_vis import plot_agent_batch_interactive
from trajdata_utils.vis import plot_agent_batch
from trajdata_utils.interactive_animation import (
    InteractiveAnimation,
    animate_agent_batch_interactive,
)
from ldm.models.diffusion.diffusion2 import GaussianDiffusion
from ldm.modules.diffusionmodules.trajunet import UNetModel
from diffuser.utils.training_ped import Trainer
import time
import copy

torch.manual_seed(0) # torch.manual_seed(5)  # torch.manual_seed(24)
# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["eupeds_eth", "eupeds_hotel", "eupeds_univ", "eupeds_zara1", "eupeds_zara2"],
    data_dirs={  # Remember to change this to match your filesystem!
        "eupeds_eth": "~/traj_dataset/eth-uty",
        "eupeds_hotel": "~/traj_dataset/eth-uty",
        "eupeds_univ": "~/traj_dataset/eth-uty",
        "eupeds_zara1": "~/traj_dataset/eth-uty",
        "eupeds_zara2": "~/traj_dataset/eth-uty",
    },
    history_sec =(0.0,  0.0), 
    future_sec =(15.9,  15.9),
    desired_dt=0.1,
    # agent_interaction_distances=defaultdict(lambda: 10.0),
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=0, # This can be set to 0 for single-threaded loading, if desired.
)

horizon = 160

unet = UNetModel(
    horizon = horizon
).cuda()

diffusion = GaussianDiffusion(
    model = unet,
    horizon = horizon
).cuda()

timestr = time.strftime("%Y%m%d-%H%M")
output_path = "./logs/eupeds/"+timestr
trainer = Trainer(
    diffusion_model=diffusion,
    data_loader=dataloader,
    results_folder=output_path
)

trainer.load(0,'./logs/eupeds/diffuser2/latest.pt')

data = iter(dataloader)
for i, batch in enumerate(dataloader):
    if i==0:
        break

batch = copy.deepcopy(batch)
batch.num_neigh = torch.tensor([1])
batch.neigh_hist = batch.neigh_hist[:,0:1,:,:]
batch.neigh_fut = batch.neigh_fut[:,0:1,:,:]

start = torch.tensor([10.0, 0.0])
goal = torch.tensor([0.0, 0.0])
batch.neigh_hist.position = start
batch.neigh_fut.position = torch.cat([torch.linspace(start[0], goal[0], 159).unsqueeze(1),
                                           torch.linspace(start[1], goal[1], 159).unsqueeze(1)], axis=1)

batch.neigh_hist.velocity = torch.tensor([0.0, 0.0])
batch.neigh_fut.velocity = torch.cat([-10/160*torch.ones(159, 1),
                                           torch.zeros(159, 1)], axis=1)

batch.neigh_hist_len = torch.tensor([1]).unsqueeze(0)
batch.neigh_fut_len = torch.tensor([159]).unsqueeze(0)
batch.neigh_hist_extents = torch.tensor([0.7500, 0.7500, 1.5000]).reshape(1,1,1,3)
batch.neigh_fut_extents = torch.tensor([0.7500, 0.7500, 1.5000]).reshape(1,1,1,3).repeat(1, 1, 159, 1)

batch.neigh_hist[:, :, :, 6:] = torch.zeros(1,1,1,2)
batch.neigh_fut[:, :, :, 6:] = torch.zeros(1,1,159,2)

start = [0, 0]
goal = [10, 0]

start = torch.tensor(start).unsqueeze(1)/10
goal = torch.tensor(goal).unsqueeze(1)/10
cond = torch.cat([start, goal], axis=1).cuda().unsqueeze(0)

vel_neigh = torch.cat((batch.neigh_hist.velocity, batch.neigh_fut.velocity), axis=2).transpose(2,3).cuda()
pos_neigh = torch.cat((batch.neigh_hist.position, batch.neigh_fut.position), axis=2).transpose(2,3).cuda()

neigh = torch.cat([pos_neigh, vel_neigh], axis=2)
neigh = neigh/10

rad = 2
rad = rad/10

traj_gen = diffusion.weight_sample(cond, neigh, rad, s_cbf=0.9, s_clf=-0.8, p=1, mode='min',
                                    apply_cbf=False,  apply_cbf_mid=False,
                                    apply_clf=True,
                                    incl_vel=True, apply_norm=True)

traj_gen[:,:2,:] = (traj_gen[:,:2,:])*10
traj_gen[:,2:4,:] = traj_gen[:,2:4,:]*10
traj_gen = traj_gen.transpose(1,2)

batch_gen = copy.deepcopy(batch)
batch_gen.agent_hist.position = traj_gen[:,0,:2].cpu()
batch_gen.agent_fut.position = traj_gen[:,1:,:2].cpu()
batch_gen.agent_hist.velocity = traj_gen[:,0,2:].cpu()
batch_gen.agent_fut.velocity = traj_gen[:,1:,2:].cpu()

gen_pos = torch.zeros(traj_gen.shape[0], traj_gen.shape[1], 2)
for i in range(1, traj_gen.shape[1]):
    gen_pos[:,i,:] = gen_pos[:,i-1,:] + traj_gen[:,i,2:].cpu()*0.1

batch_gen.agent_hist.position = gen_pos[:,0,:].cpu()
batch_gen.agent_fut.position = gen_pos[:,1:,:].cpu()

# plot_agent_batch_interactive(batch, batch_idx=0, cache_path=dataset.cache_path)
# plot_agent_batch(batch_gen, batch_idx=0)
animation = InteractiveAnimation(
            animate_agent_batch_interactive,
            batch=batch_gen,
            batch_idx=0,
            cache_path=dataset.cache_path,
        )
animation.show()