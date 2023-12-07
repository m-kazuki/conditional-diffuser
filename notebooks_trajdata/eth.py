import os
from torch.utils.data import DataLoader
from trajdata import AgentBatch, UnifiedDataset
import ipdb

# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["eupeds_eth"],
    data_dirs={  # Remember to change this to match your filesystem!
        "eupeds_eth": "~/traj_dataset/eth-uty"
    },
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
)

batch: AgentBatch
for batch in dataloader:
    # Train/evaluate/etc.
    ipdb.set_trace()
    pass