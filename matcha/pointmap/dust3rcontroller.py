# Standard imports
import os
import numpy as np
import torch

# DUSt3R-related imports
from dust3r.inference import inference as dust3r_inference
from dust3r.inference import load_model as load_dust3r_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# Custom imports
from matcha.pointmap.base import PointMap
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap
from matcha.dm_utils.model import freeze_model


class PointMapDUSt3R(PointMap):
    def __init__(
        self,
        **kwargs
    ):
        super(PointMapDUSt3R, self).__init__(**kwargs)


def get_pointmap_with_dust3r(
    # Scene data
    scene_source_path,
    n_images_in_pointmap=None,
    image_indices=None,
    white_background=False,
    eval_split=False,
    eval_split_interval=8,
    max_img_size=1600,
    pointmap_img_size=512,
    randomize_images=False,
    # DUSt3R
    dust3r_checkpoint_path:str="./dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    dust3r_global_aligner_mode:str="pc",
    dust3r_global_aligner_niter:int=300,
    dust3r_global_aligner_schedule:str='cosine',
    dust3r_global_aligner_lr:float=0.01,
    prefilter_images:str=None,
    symmetrize_pairs:bool=True,
    dust3r_batch_size:int=1,
    apply_dust3r_cleaning_at_initialization:bool=False,
    confidence_threshold:float=0.0,    
    # Misc
    device='cuda',
):
    dust3r_model = load_dust3r_model(dust3r_checkpoint_path, device)

    dir_content = os.listdir(scene_source_path)
    use_subdir = True
    for content in dir_content:
        if (
            content.endswith('.jpg') 
            or content.endswith('.png') 
            or content.endswith('.jpeg')
            or content.endswith('.JPG')
            or content.endswith('.PNG')
            or content.endswith('.JPEG')
        ):
            use_subdir = False
            print(f"Images found in {scene_source_path}. Using them for PointMap generation.")
    if use_subdir:
        source_path = os.path.join(scene_source_path, "images")
    else:
        source_path = scene_source_path            

    print("\nLoading training images...")
    data_path = os.path.join(source_path)
    all_frames_paths = sorted(os.listdir(data_path))

    img_paths = []
    if randomize_images:
        img_paths = np.random.choice(all_frames_paths, n_images_in_pointmap, replace=False)
        img_paths = [os.path.join(data_path, img_path) for img_path in img_paths]
    else:  # TODO
        if True and (dust3r_global_aligner_mode != "pair"):
            # Uniform sampling in all image
            print("\n[INFO] Sampling images with uniform distribution...\n")
            img_paths = [os.path.join(
                data_path, 
                all_frames_paths[i * (len(all_frames_paths) // (n_images_in_pointmap - 1))]
                ) for i in range(n_images_in_pointmap)]
        else:
            # Just use first images
            _interval = 1
            print("\n[WARNING] Sampling images in order...\n")
            img_paths = [os.path.join(
                data_path, 
                all_frames_paths[_interval * i]
                ) for i in range(n_images_in_pointmap)]
    img_paths = sorted(img_paths)

    images = load_images(img_paths, size=pointmap_img_size)
    original_images = load_images(img_paths, size=max_img_size)
    print(f"Images loaded.\nUsing images:")
    for img_path in img_paths:
        print(img_path)
        
    dust3r_h, dust3r_w = images[0]['img'].shape[-2], images[0]['img'].shape[-1]
    print(f"DUSt3R image size (dust3r_h x dust3r_w): {dust3r_h} x {dust3r_w}")
        
    # Freeze model
    freeze_model(dust3r_model)

    # ====================Generate PointMap with DUSt3R====================

    # Run initial DUSt3R prediction
    print("\nRun initial DUSt3R prediction...")
    with torch.no_grad():
        pairs = make_pairs(
            images, scene_graph='complete', 
            prefilter=prefilter_images, symmetrize=symmetrize_pairs
        )
        dust3r_output = dust3r_inference(pairs, dust3r_model, device, batch_size=dust3r_batch_size)

    # Compute Global Alignment
    print("\nComputing global alignment...")
    if dust3r_global_aligner_mode == "pc":
        global_aligner_mode = GlobalAlignerMode.PointCloudOptimizer
    elif dust3r_global_aligner_mode == "modular":
        global_aligner_mode = GlobalAlignerMode.ModularPointCloudOptimizer
    elif dust3r_global_aligner_mode == "pair":
        global_aligner_mode = GlobalAlignerMode.PairViewer
    else:
        raise ValueError(f"Unknown global_aligner_mode: {dust3r_global_aligner_mode}")

    # TODO: Check the following lines. It seems possible to initialize camera poses with known intrinsics and extrinsics.
    # scene.preset_pose([data[i][1].cpu().numpy() for i in range(1, 3)], [False, True, True])
    # scene.preset_intrinsics([data[i][0].cpu().numpy() for i in range(1, 3)], [False, True, True])
    dust3r_scene = global_aligner(dust3r_output, device=device, mode=global_aligner_mode)

    # Clean point cloud
    if apply_dust3r_cleaning_at_initialization:  # TODO
        print(f"\nCleaning initial point cloud...")
        dust3r_scene = dust3r_scene.clean_pointcloud()
        # dust3r_scene = dust3r_scene.clean_pointcloud(max_bad_conf=-1.)
        print("Point cloud cleaned.")

    if global_aligner_mode != GlobalAlignerMode.PairViewer:
        global_aligner_loss = dust3r_scene.compute_global_alignment(
            init="mst", 
            niter=dust3r_global_aligner_niter, 
            schedule=dust3r_global_aligner_schedule, 
            lr=dust3r_global_aligner_lr,
        )
    print("Initial prediction and global alignment done.")

    # preparing output
    print("\nPreparing DUSt3R output...")
    with torch.no_grad():
        imgs = dust3r_scene.imgs
        focals = dust3r_scene.get_focals()
        poses = dust3r_scene.get_im_poses()
        pts3d = dust3r_scene.get_pts3d()
        conf = [_conf + 0. for _conf in dust3r_scene.im_conf]
    global_dust3r_output = {
        'img_paths': img_paths,
        'images': imgs,
        'original_images': [(original_img['img'][0].permute(1, 2, 0).cpu().numpy() + 1.) / 2. for original_img in original_images],
        'focals': focals,
        'poses': poses,
        'pts3d': pts3d,
        'conf': conf,
    }

    # Compute masks
    masks=[conf > confidence_threshold for conf in global_dust3r_output['conf']]

    # Build PointMap
    scene_pm = PointMap(
        img_paths=global_dust3r_output['img_paths'], 
        images=global_dust3r_output['images'],
        original_images=global_dust3r_output['original_images'],
        focals=global_dust3r_output['focals'],
        poses=global_dust3r_output['poses'],
        points3d=global_dust3r_output['pts3d'],
        confidence=global_dust3r_output['conf'],
        masks=masks,
        device=device,
    )
    
    return scene_pm


def get_sfm_data_from_dust3r_pointmap(
    dust3r_pm:PointMapDUSt3R,
    max_sfm_points:int=200_000,
    sfm_confidence_threshold:float=3.,
    device:torch.device='cuda',
    average_focal_distances=False,
    white_background=False,
):
    n_images_in_pointmap = len(dust3r_pm.points3d)
    max_img_size = max(*dust3r_pm.images.shape[1:3])
    if max_img_size != 512:
        raise ValueError(f"Expected dust3r_pm.images to have max size of 512. Got {max_img_size}.")
    
    dust3r_cameras = CamerasWrapper(
        create_gs_cameras_from_pointmap(
            dust3r_pm,
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

    dust3r_pointmap_cameras = CamerasWrapper(
        create_gs_cameras_from_pointmap(
            dust3r_pm,
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
    
    max_sfm_points_per_frame = max_sfm_points // n_images_in_pointmap
    sfm_xyz = torch.zeros(0, 3, device=device)
    sfm_col = torch.zeros(0, 3, device=device)
    image_sfm_points = {}

    for i_frame in range(n_images_in_pointmap):
        all_frame_pts = dust3r_pm.points3d[i_frame].view(-1, 3)
        all_frame_col = dust3r_pm.images[i_frame].view(-1, 3)
        
        mask = dust3r_pm.confidence[i_frame].view(-1) > sfm_confidence_threshold
        all_frame_pts = all_frame_pts[mask]
        all_frame_col = all_frame_col[mask]
        
        sampled_idx = torch.randperm(all_frame_pts.shape[0], device=device)[:max_sfm_points_per_frame]
        
        dict_key = dust3r_pointmap_cameras.gs_cameras[i_frame].image_name.split('.')[0]
        image_sfm_points[dict_key] = len(sfm_xyz) + torch.arange(len(sampled_idx), device=device)
        
        sfm_xyz = torch.cat([sfm_xyz, all_frame_pts[sampled_idx]], dim=0)
        sfm_col = torch.cat([sfm_col, all_frame_col[sampled_idx]], dim=0)
        
    sfm_data = {
        'pointmap_cameras': dust3r_pointmap_cameras,
        'training_cameras': dust3r_cameras,
        'test_cameras': None,
        'sfm_xyz': sfm_xyz,
        'sfm_col': sfm_col,
        'image_sfm_points': image_sfm_points,
    }
    return sfm_data


def compute_dust3r_scene(
    # Data
    scene_source_path,
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
    sfm_confidence_threshold=3.,
    average_focal_distances=False,
    # DUSt3R
    dust3r_checkpoint_path="./dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    dust3r_global_aligner_mode="pc",
    dust3r_global_aligner_niter=300,
    dust3r_global_aligner_schedule='cosine',
    dust3r_global_aligner_lr=0.01,
    prefilter_images=None,
    symmetrize_pairs=True,
    dust3r_batch_size=1,
    # Misc
    device='cuda',
):
    dust3r_pm = get_pointmap_with_dust3r(
        # Scene data
        scene_source_path=scene_source_path,
        n_images_in_pointmap=n_images_in_pointmap,
        image_indices=image_indices,
        white_background=white_background,
        eval_split=eval_split,
        eval_split_interval=eval_split_interval,
        max_img_size=max_img_size,
        pointmap_img_size=pointmap_img_size,
        randomize_images=randomize_images,
        # DUSt3R
        dust3r_checkpoint_path=dust3r_checkpoint_path,
        dust3r_global_aligner_mode=dust3r_global_aligner_mode,
        dust3r_global_aligner_niter=dust3r_global_aligner_niter,
        dust3r_global_aligner_schedule=dust3r_global_aligner_schedule,
        dust3r_global_aligner_lr=dust3r_global_aligner_lr,
        prefilter_images=prefilter_images,
        symmetrize_pairs=symmetrize_pairs,
        dust3r_batch_size=dust3r_batch_size,
        apply_dust3r_cleaning_at_initialization=False,
        confidence_threshold=0.0,
        # Misc
        device=device,
    )
    
    dust3r_sfm_data = get_sfm_data_from_dust3r_pointmap(
        dust3r_pm=dust3r_pm,
        max_sfm_points=max_sfm_points,
        sfm_confidence_threshold=sfm_confidence_threshold,
        device=device,
        average_focal_distances=average_focal_distances,
        white_background=white_background,
    )
    
    return dust3r_sfm_data
