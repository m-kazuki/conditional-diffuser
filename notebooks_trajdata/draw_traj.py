import os
from torch.utils.data import DataLoader
import torch
import ipdb

from tqdm import tqdm

from trajdata import AgentBatch, AgentType, UnifiedDataset

import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

from  trajdata_utils.interactive_vis import plot_agent_batch_interactive
from trajdata_utils.vis import plot_agent_batch
from trajdata_utils.interactive_animation import (
    InteractiveAnimation,
    animate_agent_batch_interactive,
)
torch.manual_seed(6)
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
    future_sec =(16.0,  16.0),
    desired_dt=0.1,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
)

data = iter(dataloader)
batch = next(data)

# plot_agent_batch_interactive(batch, batch_idx=0, cache_path=dataset.cache_path)
pos_neigh = torch.cat((batch.agent_hist.position+batch.neigh_hist.position, batch.agent_fut.position+batch.neigh_fut.position), axis=2).transpose(2,3).cuda()
# vel_neigh1 = torch.cat((batch.agent_hist.velocity+batch.neigh_hist.velocity, batch.agent_fut.velocity+batch.neigh_fut.velocity), axis=2).transpose(2,3).cuda()

vel_neigh2 = torch.cat((batch.neigh_hist.velocity, batch.neigh_fut.velocity), axis=2).transpose(2,3).cuda()

pos = torch.cat((batch.agent_hist.position, batch.agent_fut.position), axis=1).transpose(1,2).cuda()
vel = torch.cat((batch.agent_hist.velocity, batch.agent_fut.velocity), axis=1).transpose(1,2).cuda()


ipdb.set_trace()

animation = InteractiveAnimation(
            animate_agent_batch_interactive,
            batch=batch,
            batch_idx=0,
            cache_path=dataset.cache_path,
        )
animation.show()