# Standard imports
import os
import numpy as np
import torch
import json
from PIL import Image

# DUSt3R imports
# from dust3r.utils.image import load_images
from matcha.dm_utils.dust3r_image import load_images

# Custom imports
from .base import PointMap
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap
from matcha.dm_utils.model import freeze_model


class PointMapMASt3R(PointMap):
    def __init__(
        self,
        **kwargs
    ):
        super(PointMapMASt3R, self).__init__(**kwargs)
        
        
def load_mast3r_matches(npz_path, height_first=True):
    """
    Load matches from a .npz file and convert them to tensors.
    
    Args:
        npz_path: Path to the .npz file containing matches
        
    Returns:
        match_to_img: Tensor of shape (n_match, 2) containing image indices for each match
        match_to_pix: Tensor of shape (n_match, 2, 2) containing UV coordinates for each match
        idx_to_image: Dictionary mapping image indices to image names
    """
    
    # Example usage:
    # match_to_img, match_to_pix = load_matches_as_tensors('matches.npz')
    # print(f"Number of matches: {len(match_to_img)}")
    # print(f"match_to_img shape: {match_to_img.shape}")  # (n_match, 2)
    # print(f"match_to_pix shape: {match_to_pix.shape}")  # (n_match, 2, 2)
    
    # Load the matches
    data = np.load(npz_path, allow_pickle=True)
    
    # Create a mapping from image names to indices
    all_images = sorted(list(set(
        name for pair in data.keys() 
        for name in pair.split('___')
    )))
    image_to_idx = {name: idx for idx, name in enumerate(all_images)}
    idx_to_image = {idx: name for idx, name in enumerate(all_images)}
    
    # Initialize lists to store matches
    all_img_indices = []
    all_pixels = []

    # Process each pair
    for pair_key, matches in data.items():
        img1_name, img2_name = pair_key.split('___')
        img1_idx = image_to_idx[img1_name]
        img2_idx = image_to_idx[img2_name]
        
        # Get matches for this pair
        matches_im0 = matches.item()['matches_im0']
        matches_im1 = matches.item()['matches_im1']
        n_matches = len(matches_im0)
        
        # Add image indices for each match
        pair_indices = np.array([[img1_idx, img2_idx]] * n_matches)
        all_img_indices.append(pair_indices)
        
        # Stack pixel coordinates
        pair_pixels = np.stack([matches_im0, matches_im1], axis=1)
        all_pixels.append(pair_pixels)

    # Concatenate all matches
    match_to_img = torch.from_numpy(np.concatenate(all_img_indices, axis=0))
    match_to_pix = torch.from_numpy(np.concatenate(all_pixels, axis=0))
    
    if height_first:
        match_to_pix = match_to_pix[:, :, [1, 0]]
    
    return match_to_img, match_to_pix, idx_to_image


def select_match_values(
    values:torch.Tensor, 
    match_to_img:torch.Tensor, 
    match_to_pix:torch.Tensor, 
    height_first:bool=True
):
    """Select values from a map based on match indices.

    Args:
        values (torch.Tensor): Map of values to select from. 
            Should have shape (N_images, H, W, ...).
        match_to_img (torch.Tensor): Tensor of shape (M, 2) containing the indices 
            of the first values to select from the map for each match. 
            Should be in the range [0, N_images).
        match_to_pix (torch.Tensor): Tensor of shape (M, 2, 2) containing the pixel 
            coordinates of the first values to select from the map for each match.
            Should be in the range [0, H) x [0, W) if height_first is True, 
            or [0, W) x [0, H) otherwise.
        height_first (bool): Whether the first dimension of the match_to_pix tensor 
            is the height or the width.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Tensor of shape (M, ...) containing the selected values for the first 
              image in each match.
            - Tensor of shape (M, ...) containing the selected values for the second 
              image in each match.
    """
    N, H, W = values.shape[:3]
    value_shape = values.shape[3:]
    
    if height_first:
        value_0 = values.view(-1, *value_shape)[match_to_img[:, 0] * H * W + match_to_pix[:, 0, 0] * W + match_to_pix[:, 0, 1]]
        value_1 = values.view(-1, *value_shape)[match_to_img[:, 1] * H * W + match_to_pix[:, 1, 0] * W + match_to_pix[:, 1, 1]]
    else:
        value_0 = values.view(-1, *value_shape)[match_to_img[:, 0] * H * W + match_to_pix[:, 0, 1] * W + match_to_pix[:, 0, 0]]
        value_1 = values.view(-1, *value_shape)[match_to_img[:, 1] * H * W + match_to_pix[:, 1, 1] * W + match_to_pix[:, 1, 0]]        
    
    return value_0, value_1


def get_minimal_projections_diffs(
    points3d:torch.Tensor, 
    cameras:CamerasWrapper,
    match_to_img:torch.Tensor, 
    match_to_pix:torch.Tensor, 
    height_first:bool=True,
    znear:float=1e-6,
    loss_power:float=0.5,
):
    """Get the minimal projections diffs for a MASt3R pointmap and the corresponding 
    set of matches.

    Args:
        points3d (torch.Tensor): Tensor of shape (N, H, W, 3) containing the pointmaps.
        cameras (CamerasWrapper): CamerasWrapper object containing the cameras.
        match_to_img (torch.Tensor): Tensor of shape (M, 2) containing the indices 
            of the first values to select from the map for each match.
        match_to_pix (torch.Tensor): Tensor of shape (M, 2, 2) containing the pixel 
            coordinates of the first values to select from the map for each match.
        height_first (bool, optional): Whether the first dimension of the match_to_pix 
            tensor is the height or the width. Defaults to True.
        znear (float, optional): Near clipping plane. Defaults to 1e-6.
        loss_power (float, optional): Power to raise the minimal projections diffs to. 
            Defaults to 0.5.

    Returns:
        torch.Tensor: Tensor of shape (M, ) containing the minimal projections diffs.
    """
    # Get 3D points for matches
    match_to_3D_0, match_to_3D_1 = select_match_values(points3d, match_to_img, match_to_pix, height_first)

    # Not a good idea to try to minimize distance between 3D points, as the 3D point is not necessarily seen in both images.
    # We only know about their projections.
    # So let's try to minimize distance between projections!
    full_projection_matrices = torch.cat([gs_camera.full_proj_transform[None] for gs_camera in cameras.gs_cameras], dim=0)
    match_projection_matrices_0 = full_projection_matrices[match_to_img[:, 0]]
    match_projection_matrices_1 = full_projection_matrices[match_to_img[:, 1]]
    
    projections_0_to_0 = torch.nn.functional.pad(match_to_3D_0[:, None, :], (0, 1), value=1.) @ match_projection_matrices_0
    projections_1_to_0 = torch.nn.functional.pad(match_to_3D_1[:, None, :], (0, 1), value=1.) @ match_projection_matrices_0
    projections_0_to_1 = torch.nn.functional.pad(match_to_3D_0[:, None, :], (0, 1), value=1.) @ match_projection_matrices_1
    projections_1_to_1 = torch.nn.functional.pad(match_to_3D_1[:, None, :], (0, 1), value=1.) @ match_projection_matrices_1

    all_projections = torch.cat([projections_0_to_0, projections_1_to_0, projections_0_to_1, projections_1_to_1], dim=1)  # (M, n_projections, 4)
    invalid_projections = (all_projections[..., 3] < znear).any(dim=-1)  # (M, )
    all_projections = all_projections[~invalid_projections]  # (M_valid, n_projections, 4)
    all_projections = all_projections[..., :2] / all_projections[..., 3:4]  # (M_valid, n_projections, 2)

    projections_diffs = torch.cat(
        [
            (all_projections[:, 0] - all_projections[:, 1]).norm(dim=-1, keepdim=True),  # (M_valid, 1)
            (all_projections[:, 2] - all_projections[:, 3]).norm(dim=-1, keepdim=True),  # (M_valid, 1)
        ], dim=1
    )  # (M_valid, 2)

    min_projections_diffs = projections_diffs.min(dim=-1).values  # (M_valid, )
    
    return min_projections_diffs ** loss_power
        

def get_pointmap_with_mast3r(
    # Scene data
    mast3r_scene_source_path,
    n_images_in_pointmap=None,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    confidence_threshold:float=0.0,
    # Misc
    device='cuda',
):
    print("[WARNING] This function returns a pointmap given a precomputed scene output by MASt3R.")
    # TODO: Make it possible to take a subset of the images. This is useful for a MASt3R scene computed with dense views.
    
    # Load the cameras file
    print(f"Loading camera file from {mast3r_scene_source_path}...")
    cameras_file = os.path.join(mast3r_scene_source_path, 'cameras.json')
    with open(cameras_file, 'r') as file:
        cameras = json.load(file)
    n_total_images = len(cameras['filepaths'])
    if n_images_in_pointmap is None:
        n_images_in_pointmap = n_total_images
    if image_indices is None:
        image_indices = [i * (len(cameras['filepaths']) // (n_images_in_pointmap - 1)) for i in range(n_images_in_pointmap)]
    else:
        n_images_in_pointmap = len(image_indices)
        
    _img_paths = [cameras['filepaths'][i] for i in image_indices]
    _focals = [[cameras['focals'][i]] for i in image_indices]
    _poses = [cameras['cams2world'][i] for i in image_indices]
    print("Indices of the images in the pointmap:", image_indices)
    
    # Fix filepaths
    _img_file_names = [os.path.basename(img_path) for img_path in _img_paths]
    _img_paths = [os.path.join(mast3r_scene_source_path, 'images', img_file_name) for img_file_name in _img_file_names]
    
    # Load the pointmap files
    pointmaps_dir = os.path.join(mast3r_scene_source_path, 'pointmaps')
    all_pointmaps_files = sorted([os.path.join(pointmaps_dir, f) for f in os.listdir(pointmaps_dir) if f.endswith('.json')])
    pointmaps_files = [all_pointmaps_files[i * (len(all_pointmaps_files) // (n_images_in_pointmap - 1))] for i in range(n_images_in_pointmap)]
    print(f"Loading {len(pointmaps_files)} pointmaps from {mast3r_scene_source_path}...")
    scene = {'rgb': [], 'points': [], 'confs': [],}
    for pointmap_file in pointmaps_files:
        with open(pointmap_file, 'r') as file:
            pointmap_dir = json.load(file)
        scene['rgb'].append(pointmap_dir['rgb'])
        scene['points'].append(pointmap_dir['points'])
        scene['confs'].append(pointmap_dir['confs'])
    
    # Build PointMap
    if scene['rgb'][0] is None:
        _images = [((img['img'][0].permute(1, 2, 0) + 1.) / 2).numpy() for img in load_images(_img_paths, size=pointmap_img_size)]
    else:
        _images = scene['rgb']
    
    # Load images with PIL
    _original_images = [Image.open(img_path)for img_path in _img_paths]
    _original_images_height = _original_images[0].height
    _original_images_width = _original_images[0].width
    resize_factor = max_img_size / max(_original_images_height, _original_images_width)

    # Resize so that the largest side is max_img_size
    _original_images = [np.array(
        img.resize((int(round(_original_images_width * resize_factor)), int(round(_original_images_height * resize_factor))), Image.LANCZOS)
    ) / 255. for img in _original_images]    
    _points3d = [np.array(pts).reshape(np.array(_images)[0].shape) for pts in scene['points']]
    
    # Build PointMap
    scene_pm = PointMap(
        img_paths=_img_paths,
        images=_images,
        original_images=_original_images,
        focals=_focals,
        poses=_poses,
        points3d=_points3d,
        confidence=scene['confs'],
        masks=[np.array(conf) > confidence_threshold for conf in scene['confs']],
        device=device,
    )
    
    if (
        scene_pm._images.dtype == np.dtype(np.float64)
        or scene_pm._original_images.dtype == np.dtype(np.float64)
        or scene_pm._focals.dtype == np.dtype(np.float64)
        or scene_pm._poses.dtype == np.dtype(np.float64)
        or scene_pm._points3d.dtype == np.dtype(np.float64)
        or scene_pm._confidence.dtype == np.dtype(np.float64)
    ):
        print("[WARNING] PointMap data types are float64. Converting to float32.")
        scene_pm._images = scene_pm._images.astype(np.float32)
        scene_pm._original_images = scene_pm._original_images.astype(np.float32)
        scene_pm._focals = scene_pm._focals.astype(np.float32)
        scene_pm._poses = scene_pm._poses.astype(np.float32)
        scene_pm._points3d = scene_pm._points3d.astype(np.float32)
        scene_pm._confidence = scene_pm._confidence.astype(np.float32)
    
    return scene_pm
    
    
def get_sfm_data_from_mast3r_pointmap(
    mast3r_pm:PointMapMASt3R,
    max_sfm_points:int=200_000,
    sfm_confidence_threshold:float=3.,
    device:torch.device='cuda',
    average_focal_distances=False,
    white_background=False,
):
    n_images_in_pointmap = len(mast3r_pm.points3d)
    max_img_size = max(*mast3r_pm.images.shape[1:3])
    if max_img_size != 512:
        raise ValueError(f"Expected mast3r_pm.images to have max size of 512. Got {max_img_size}.")
    
    mast3r_cameras = CamerasWrapper(
        create_gs_cameras_from_pointmap(
            mast3r_pm,
            image_resolution=1, 
            load_gt_images=True, 
            max_img_size=max_img_size, 
            use_original_image_size=True,
            white_background=white_background,  # TODO: Implement white background
            average_focal_distances=average_focal_distances,
            verbose=False,
        ), 
        no_p3d_cameras=False
    )

    mast3r_pointmap_cameras = CamerasWrapper(
        create_gs_cameras_from_pointmap(
            mast3r_pm,
            image_resolution=1, 
            load_gt_images=True, 
            max_img_size=max_img_size, 
            use_original_image_size=False,
            white_background=white_background,
            average_focal_distances=average_focal_distances,
            verbose=False,
        ),
        no_p3d_cameras=False
    )
    
    if (max_sfm_points is None) or (max_sfm_points == -1):
        max_sfm_points_per_frame = 'none'
    else:
        max_sfm_points_per_frame = max_sfm_points // n_images_in_pointmap
    sfm_xyz = torch.zeros(0, 3, device=device)
    sfm_col = torch.zeros(0, 3, device=device)
    image_sfm_points = {}

    for i_frame in range(n_images_in_pointmap):
        all_frame_pts = mast3r_pm.points3d[i_frame].view(-1, 3)
        all_frame_col = mast3r_pm.images[i_frame].view(-1, 3)
        
        mask = mast3r_pm.confidence[i_frame].view(-1) > sfm_confidence_threshold
        all_frame_pts = all_frame_pts[mask]
        all_frame_col = all_frame_col[mask]
        
        if max_sfm_points is None:
            sampled_idx = torch.arange(all_frame_pts.shape[0], device=device)
        else:
            sampled_idx = torch.randperm(all_frame_pts.shape[0], device=device)[:max_sfm_points_per_frame]
        
        dict_key = mast3r_pointmap_cameras.gs_cameras[i_frame].image_name.split('.')[0]
        image_sfm_points[dict_key] = len(sfm_xyz) + torch.arange(len(sampled_idx), device=device)
        
        sfm_xyz = torch.cat([sfm_xyz, all_frame_pts[sampled_idx]], dim=0)
        sfm_col = torch.cat([sfm_col, all_frame_col[sampled_idx]], dim=0)
        
    sfm_data = {
        'pointmap_cameras': mast3r_pointmap_cameras,
        'training_cameras': mast3r_cameras,
        'test_cameras': None,
        'sfm_xyz': sfm_xyz,
        'sfm_col': sfm_col,
        'image_sfm_points': image_sfm_points,
    }
    return sfm_data


def compute_mast3r_scene(
    # Data
    mast3r_scene_source_path,
    n_images_in_pointmap,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    # Output
    max_sfm_points=200_000,
    sfm_confidence_threshold=3.,  # Should it be 0.?
    average_focal_distances=False,
    # Misc
    device='cuda',
    return_pointmap=False,
):
        
    mast3r_pm = get_pointmap_with_mast3r(
        # Scene data
        mast3r_scene_source_path=mast3r_scene_source_path,
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        white_background=white_background,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        max_img_size=max_img_size,
        pointmap_img_size=pointmap_img_size,
        randomize_images=randomize_images,
        # DUSt3R
        confidence_threshold=0.0,
        # Misc
        device=device,
    )
    
    mast3r_sfm_data = get_sfm_data_from_mast3r_pointmap(
        mast3r_pm=mast3r_pm,
        max_sfm_points=max_sfm_points,
        sfm_confidence_threshold=sfm_confidence_threshold,
        device=device,
        average_focal_distances=average_focal_distances,
        white_background=white_background,
    )
    
    if return_pointmap:
        return mast3r_sfm_data, mast3r_pm
    return mast3r_sfm_data
