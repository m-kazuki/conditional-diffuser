from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.transform import rotate
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from trajdata.augmentation import BatchAugmentation
from diffuser.datasets.batch_trajdata import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.state import TORCH_STATE_TYPES
from trajdata.maps import VectorMap
from trajdata.utils import arr_utils


class CustomCollateData:
    @staticmethod
    def __collate__(elements: list) -> any:
        raise NotImplementedError

    def __to__(self, device, non_blocking=False):
        # Example for moving all elements of a list to a device:
        # return LanesList([[pts.to(device, non_blocking=non_blocking)
        #           for pts in lanelist] for lanelist in self])
        raise NotImplementedError


def _collate_data(elems):
    if hasattr(elems[0], "__collate__"):
        return elems[0].__collate__(elems)
    else:
        return torch.as_tensor(np.stack(elems))


def raster_map_collate_fn_agent(
    batch_elems: List[AgentBatchElement],
):
    if batch_elems[0].map_patch is None:
        return None, None, None, None

    map_names = [batch_elem.map_name for batch_elem in batch_elems]

    # Ensuring that any empty map patches have the correct number of channels
    # prior to collation.
    has_data: np.ndarray = np.array(
        [batch_elem.map_patch.has_data for batch_elem in batch_elems],
        dtype=bool,
    )
    no_data: np.ndarray = ~has_data

    patch_channels: np.ndarray = np.array(
        [batch_elem.map_patch.data.shape[0] for batch_elem in batch_elems],
        dtype=int,
    )

    desired_num_channels: int
    if np.any(has_data):
        # If any of the batch elements' maps have data, then use
        # their number of channels as the reference.
        unique_num_channels = np.unique(patch_channels[has_data])
    else:
        # All map patches in this batch are from datasets with no maps.
        unique_num_channels = np.unique(patch_channels)

    if unique_num_channels.size > 1:
        raise ValueError(
            "Maps must all have the same number of channels in a batch, "
            f"but found maps with {unique_num_channels.tolist()} channels."
        )

    desired_num_channels = unique_num_channels[0].item()

    # Getting the map patch data and preparing it for batched rotation.
    patch_size_y, patch_size_x = batch_elems[0].map_patch.data.shape[-2:]
    patch_data: Tensor = torch.empty(
        (len(batch_elems), desired_num_channels, patch_size_y, patch_size_x)
    )

    if np.any(has_data):
        patch_data[has_data] = torch.as_tensor(
            np.stack(
                [
                    batch_elem.map_patch.data
                    for idx, batch_elem in enumerate(batch_elems)
                    if has_data[idx]
                ]
            ),
            dtype=torch.float,
        )

    if np.any(no_data):
        patch_data[no_data] = torch.as_tensor(
            np.stack(
                [
                    batch_elem.map_patch.data
                    for idx, batch_elem in enumerate(batch_elems)
                    if no_data[idx]
                ]
            ),
            dtype=torch.float,
        ).expand(-1, desired_num_channels, -1, -1)

    patch_size: int = batch_elems[0].map_patch.crop_size
    assert all(
        batch_elem.map_patch.crop_size == patch_size for batch_elem in batch_elems
    )

    rot_angles: Tensor = torch.as_tensor(
        [batch_elem.map_patch.rot_angle for batch_elem in batch_elems],
        dtype=torch.float,
    )
    resolution: Tensor = torch.as_tensor(
        [batch_elem.map_patch.resolution for batch_elem in batch_elems],
        dtype=torch.float,
    )
    rasters_from_world_tf: Tensor = torch.as_tensor(
        np.stack(
            [batch_elem.map_patch.raster_from_world_tf for batch_elem in batch_elems]
        ),
        dtype=torch.float,
    )

    center_y: int = patch_size_y // 2
    center_x: int = patch_size_x // 2
    half_extent: int = patch_size // 2

    if (
        torch.count_nonzero(rot_angles) == 0
        and patch_size == patch_data.shape[-1] == patch_data.shape[-2]
    ):
        rasters_from_world_tf = torch.bmm(
            torch.tensor(
                [
                    [
                        [1.0, 0.0, half_extent],
                        [0.0, 1.0, half_extent],
                        [0.0, 0.0, 1.0],
                    ]
                ],
                dtype=rasters_from_world_tf.dtype,
                device=rasters_from_world_tf.device,
            ).expand((rasters_from_world_tf.shape[0], -1, -1)),
            rasters_from_world_tf,
        )

        rot_crop_patches: Tensor = patch_data

    else:
        # Batch rotating patches by rot_angles.
        rot_patches: Tensor = rotate(patch_data, torch.rad2deg(rot_angles))

        # Center cropping via slicing.
        rot_crop_patches: Tensor = rot_patches[
            ...,
            center_y - half_extent : center_y + half_extent,
            center_x - half_extent : center_x + half_extent,
        ]

        rasters_from_world_tf = torch.bmm(
            arr_utils.transform_matrices(
                -rot_angles,
                torch.tensor([[half_extent, half_extent]]).expand(
                    (rot_angles.shape[0], -1)
                ),
            ),
            rasters_from_world_tf,
        )

    return (
        map_names,
        rot_crop_patches,
        resolution,
        rasters_from_world_tf,
    )


def raster_map_collate_fn_scene(
    batch_elems: List[SceneBatchElement],
    max_agent_num: Optional[int] = None,
    pad_value: Any = np.nan,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:

    if batch_elems[0].map_patches is None:
        return None, None, None, None

    patch_size: int = batch_elems[0].map_patches[0].crop_size
    assert all(
        batch_elem.map_patches[0].crop_size == patch_size for batch_elem in batch_elems
    )

    map_names: List[str] = list()
    num_agents: List[int] = list()
    agents_rasters_from_world_tfs: List[np.ndarray] = list()
    agents_patches: List[np.ndarray] = list()
    agents_rot_angles_list: List[float] = list()
    agents_res_list: List[float] = list()

    for elem in batch_elems:
        map_names.append(elem.map_name)
        num_agents.append(min(elem.num_agents, max_agent_num))
        agents_rasters_from_world_tfs += [
            x.raster_from_world_tf for x in elem.map_patches[:max_agent_num]
        ]
        agents_patches += [x.data for x in elem.map_patches[:max_agent_num]]
        agents_rot_angles_list += [
            x.rot_angle for x in elem.map_patches[:max_agent_num]
        ]
        agents_res_list += [x.resolution for x in elem.map_patches[:max_agent_num]]

    patch_data: Tensor = torch.as_tensor(np.stack(agents_patches), dtype=torch.float)
    agents_rot_angles: Tensor = torch.as_tensor(
        np.stack(agents_rot_angles_list), dtype=torch.float
    )
    agents_rasters_from_world_tf: Tensor = torch.as_tensor(
        np.stack(agents_rasters_from_world_tfs), dtype=torch.float
    )
    agents_resolution: Tensor = torch.as_tensor(
        np.stack(agents_res_list), dtype=torch.float
    )

    patch_size_y, patch_size_x = patch_data.shape[-2:]
    center_y: int = patch_size_y // 2
    center_x: int = patch_size_x // 2
    half_extent: int = patch_size // 2

    if torch.count_nonzero(agents_rot_angles) == 0:
        agents_rasters_from_world_tf = torch.bmm(
            torch.tensor(
                [
                    [
                        [1.0, 0.0, half_extent],
                        [0.0, 1.0, half_extent],
                        [0.0, 0.0, 1.0],
                    ]
                ],
                dtype=agents_rasters_from_world_tf.dtype,
                device=agents_rasters_from_world_tf.device,
            ).expand((agents_rasters_from_world_tf.shape[0], -1, -1)),
            agents_rasters_from_world_tf,
        )

        rot_crop_patches = patch_data
    else:
        agents_rasters_from_world_tf = torch.bmm(
            arr_utils.transform_matrices(
                -agents_rot_angles,
                torch.tensor([[half_extent, half_extent]]).expand(
                    (agents_rot_angles.shape[0], -1)
                ),
            ),
            agents_rasters_from_world_tf,
        )

        # Batch rotating patches by rot_angles.
        rot_patches: Tensor = rotate(patch_data, torch.rad2deg(agents_rot_angles))

        # Center cropping via slicing.
        rot_crop_patches = rot_patches[
            ...,
            center_y - half_extent : center_y + half_extent,
            center_x - half_extent : center_x + half_extent,
        ]

    rot_crop_patches = split_pad_crop(
        rot_crop_patches, num_agents, pad_value=pad_value, desired_size=max_agent_num
    )

    agents_rasters_from_world_tf = split_pad_crop(
        agents_rasters_from_world_tf,
        num_agents,
        pad_value=pad_value,
        desired_size=max_agent_num,
    )
    agents_resolution = split_pad_crop(
        agents_resolution, num_agents, pad_value=0, desired_size=max_agent_num
    )

    return map_names, rot_crop_patches, agents_resolution, agents_rasters_from_world_tf


def agent_collate_fn(
    batch_elems: List[AgentBatchElement],
    return_dict: bool,
    pad_format: str,
    batch_augments: Optional[List[BatchAugmentation]] = None,
) -> Union[AgentBatch, Dict[str, Any]]:
    batch_size: int = len(batch_elems)
    history_pad_dir: arr_utils.PadDirection = (
        arr_utils.PadDirection.BEFORE
        if pad_format == "outside"
        else arr_utils.PadDirection.AFTER
    )

    data_index_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    scene_ts_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    dt_t: Tensor = torch.zeros((batch_size,), dtype=torch.float)
    agent_type_t: Tensor = torch.zeros((batch_size,), dtype=torch.int)
    agent_names: List[str] = list()

    # get agent state and obs format from first item in list
    state_format = batch_elems[0].curr_agent_state_np._format
    obs_format = batch_elems[0].cache.obs_type._format
    AgentStateTensor = TORCH_STATE_TYPES[state_format]
    AgentObsTensor = TORCH_STATE_TYPES[obs_format]

    curr_agent_state: List[AgentStateTensor] = list()

    agent_history: List[AgentObsTensor] = list()
    agent_history_extent: List[Tensor] = list()
    agent_history_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)

    agent_future: List[AgentObsTensor] = list()
    agent_future_extent: List[Tensor] = list()
    agent_future_len: Tensor = torch.zeros((batch_size,), dtype=torch.long)


    elem: AgentBatchElement
    for idx, elem in enumerate(batch_elems):
        data_index_t[idx] = elem.data_index
        scene_ts_t[idx] = elem.scene_ts
        dt_t[idx] = elem.dt
        agent_names.append(elem.agent_name)
        agent_type_t[idx] = elem.agent_type.value

        curr_agent_state.append(
            torch.as_tensor(elem.curr_agent_state_np, dtype=torch.float)
        )

        agent_history.append(
            arr_utils.convert_with_dir(
                elem.agent_history_np,
                dtype=torch.float,
                time_dim=-2,
                pad_dir=history_pad_dir,
            )
        )
        agent_history_extent.append(
            arr_utils.convert_with_dir(
                elem.agent_history_extent_np,
                dtype=torch.float,
                time_dim=-2,
                pad_dir=history_pad_dir,
            )
        )
        agent_history_len[idx] = elem.agent_history_len

        agent_future.append(torch.as_tensor(elem.agent_future_np, dtype=torch.float))
        agent_future_extent.append(
            torch.as_tensor(elem.agent_future_extent_np, dtype=torch.float)
        )
        agent_future_len[idx] = elem.agent_future_len

    curr_agent_state_t: AgentStateTensor = torch.stack(curr_agent_state).as_subclass(
        AgentStateTensor
    )

    agent_history_t: AgentObsTensor = arr_utils.pad_with_dir(
        agent_history,
        time_dim=-2,
        pad_dir=history_pad_dir,
        batch_first=True,
        padding_value=np.nan,
    ).as_subclass(AgentObsTensor)
    agent_history_extent_t: Tensor = arr_utils.pad_with_dir(
        agent_history_extent,
        time_dim=-2,
        pad_dir=history_pad_dir,
        batch_first=True,
        padding_value=np.nan,
    )

    agent_future_t: AgentObsTensor = pad_sequence(
        agent_future, batch_first=True, padding_value=np.nan
    ).as_subclass(AgentObsTensor)
    agent_future_extent_t: Tensor = pad_sequence(
        agent_future_extent, batch_first=True, padding_value=np.nan
    )

    # Padding history/future in case the length is less than
    # the minimum desired history/future length.
    if elem.history_sec[0] is not None:
        hist_len = int(elem.history_sec[0] / elem.dt) + 1
        if agent_history_t.shape[-2] < hist_len:
            to_add: int = hist_len - agent_history_t.shape[-2]
            agent_history_t = F.pad(
                agent_history_t,
                (0, 0, to_add, 0)
                if history_pad_dir == arr_utils.PadDirection.BEFORE
                else (0, 0, 0, to_add),
                value=np.nan,
            ).as_subclass(AgentObsTensor)

        if agent_history_extent_t.shape[-2] < hist_len:
            to_add: int = hist_len - agent_history_extent_t.shape[-2]
            agent_history_extent_t = F.pad(
                agent_history_extent_t,
                (0, 0, to_add, 0)
                if history_pad_dir == arr_utils.PadDirection.BEFORE
                else (0, 0, 0, to_add),
                value=np.nan,
            )

    if elem.future_sec[0] is not None:
        fut_len = int(elem.future_sec[0] / elem.dt)
        if agent_future_t.shape[-2] < fut_len:
            agent_future_t = F.pad(
                agent_future_t,
                (0, 0, 0, fut_len - agent_future_t.shape[-2]),
                value=np.nan,
            ).as_subclass(AgentObsTensor)

        if agent_future_extent_t.shape[-2] < fut_len:
            agent_future_extent_t = F.pad(
                agent_future_extent_t,
                (0, 0, 0, fut_len - agent_future_extent_t.shape[-2]),
                value=np.nan,
            )


    agents_from_world_tf = torch.as_tensor(
        np.stack([batch_elem.agent_from_world_tf for batch_elem in batch_elems]),
        dtype=torch.float,
    )

    scene_ids = [batch_elem.scene_id for batch_elem in batch_elems]

    extras: Dict[str, Tensor] = {}
    for key in batch_elems[0].extras.keys():
        extras[key] = _collate_data(
            [batch_elem.extras[key] for batch_elem in batch_elems]
        )

    batch = AgentBatch(
        data_idx=data_index_t,
        scene_ts=scene_ts_t,
        dt=dt_t,
        agent_name=agent_names,
        agent_type=agent_type_t,
        curr_agent_state=curr_agent_state_t,
        agent_hist=agent_history_t,
        agent_hist_extent=agent_history_extent_t,
        agent_hist_len=agent_history_len,
        agent_fut=agent_future_t,
        agent_fut_extent=agent_future_extent_t,
        agent_fut_len=agent_future_len,
        agents_from_world_tf=agents_from_world_tf,
        scene_ids=scene_ids,
        history_pad_dir=history_pad_dir,
        extras=extras,
    )

    if batch_augments:
        for batch_aug in batch_augments:
            batch_aug.apply_agent(batch)

    if return_dict:
        return asdict(batch)

    return batch


def split_pad_crop(
    batch_tensor, sizes, pad_value: float = 0.0, desired_size: Optional[int] = None
) -> Tensor:
    """Split a batched tensor into different sizes and pad them to the same size

    Args:
        batch_tensor: tensor in bach or split tensor list
        sizes (torch.Tensor): sizes of each entry
        pad_value (float, optional): padding value. Defaults to 0.0
        desired_size (int, optional): desired size. Defaults to None.
    """

    if isinstance(batch_tensor, Tensor):
        x = torch.split(batch_tensor, sizes)
        cat_fun = torch.cat
        full_fun = torch.full
    elif isinstance(batch_tensor, np.ndarray):
        x = np.split(batch_tensor, sizes)
        cat_fun = np.concatenate
        full_fun = np.full
    elif isinstance(batch_tensor, List):
        # already splitted in list
        x = batch_tensor
        if isinstance(batch_tensor[0], Tensor):
            cat_fun = torch.cat
            full_fun = torch.full
        elif isinstance(batch_tensor[0], np.ndarray):
            cat_fun = np.concatenate
            full_fun = np.full
    else:
        raise ValueError("wrong data type for batch tensor")

    x: Tensor = pad_sequence(x, batch_first=True, padding_value=pad_value)
    if desired_size is not None:
        if x.shape[1] >= desired_size:
            x = x[:, :desired_size]
        else:
            bs, max_size = x.shape[:2]
            x = cat_fun(
                (x, full_fun([bs, desired_size - max_size, *x.shape[2:]], pad_value)), 1
            )

    return x

