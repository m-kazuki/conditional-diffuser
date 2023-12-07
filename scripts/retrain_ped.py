import diffuser.utils as utils
import ipdb
from trajdata import AgentBatch, AgentType, UnifiedDataset
from torch.utils.data import DataLoader
import os
from diffuser.utils.training_ped import Trainer
from ldm.models.diffusion.ddpm import Diffusion
import time


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#


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
    desired_dt=0.1
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=0, # This can be set to 0 for single-threaded loading, if desired.
)



#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#


diffusion = Diffusion(
    horizon = 160
).cuda()

output_path = "./logs/eupeds/20231114-1826/"

trainer = Trainer(
    diffusion_model=diffusion,
    data_loader=dataloader,
    results_folder=output_path
)

trainer.load(0, output_path+"latest.pt")

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

