import os
import numpy as np
import torch

from matcha.dm_scene.parallel_aligner import (
    ParallelAligner,
    MLPParams,
    ChartsEncodingParams,
    DepthEncodingParams,
    MultiResChartsEncodingParams,
)
from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps
from matcha.dm_scene.cameras import CamerasWrapper, create_gs_cameras_from_pointmap, rescale_cameras
from matcha.pointmap.mast3rcontroller import load_mast3r_matches


# TODO: Update the default values of the parameters
def align_charts_in_parallel(
    # Scene
    scene_pm,
    # Data parameters
    reference_data,
    masks=None,
    rendering_size=1600,
    target_scale=5.,
    # ParallelAligner architecture parameters
    use_learnable_depth_encoding=True,
    learnable_depth_encoding_mode='add',
    predict_in_disparity_space=False,
    use_learnable_confidence=True,
    use_meta_mlp=False,
    use_lora_mlp=False,
    lora_rank=4,
    n_lora_layers=2,
    use_multi_res_charts_encoding=True,
    # ParallelAligner optimization parameters
    n_iterations=1000,
    use_gradient_loss=False,
    use_hessian_loss=False,
    use_normal_loss=True,
    use_curvature_loss=True,
    use_matching_loss=True,
    use_reprojection_loss=False,
    matching_thr_factor=1./20.,
    matching_update_iters=None,
    use_confidence_in_matching_loss=False,
    weight_encodings_with_confidence=False,
    regularize_chart_encodings_norms=False,
    use_total_variation_on_depth_encodings=False,
    gradient_loss_weight=50.,
    hessian_loss_weight=100.,
    normal_loss_weight=4.,
    curvature_loss_weight=1.,
    reprojection_loss_weight=2.,
    reprojection_loss_power=0.5,
    reprojection_matches_file=None,
    matching_loss_weight=5.,
    chart_encodings_norm_loss_weight=2.,
    total_variation_on_depth_encodings_weight=5.0,
    encodings_lr=1e-2,
    mlp_lr=1e-3,
    confidence_lr=1e-3,
    lr_update_iters=[1000],
    lr_update_factor=0.1,
    verbose=True,
    return_training_losses=False,
    save_charts_data=True,
    charts_data_path='./',
):
    device = scene_pm.points3d.device
    
    charts_encoding_params=ChartsEncodingParams()
    depth_encoding_params=DepthEncodingParams()
    mlp_params=MLPParams()
    
    if verbose:
        print("===== ParallelAligner parameters =====\n")
        print("Charts encoding dim", charts_encoding_params.encoding_dim)
        print("Charts encoding resolution factor", charts_encoding_params.resolution_factor)
        print("Charts encoding initialization range", charts_encoding_params.initialization_range, '\n')
        print("Depth encoding dim", depth_encoding_params.encoding_dim)
        print("Depth encoding n bins", depth_encoding_params.n_bins)
        print("Depth encoding initialization range", depth_encoding_params.initialization_range, '\n')
        print("MLP input dim", mlp_params.n_deformation_layers)
        print("MLP deformation layer size", mlp_params.deformation_layer_size, '\n')
    
    # Build cameras
    cam_list = create_gs_cameras_from_pointmap(
        scene_pm,
        image_resolution=1, 
        load_gt_images=True, 
        max_img_size=rendering_size, 
        use_original_image_size=True,
        average_focal_distances=False,
        verbose=False,
    )
    pointmap_cameras = CamerasWrapper(cam_list, no_p3d_cameras=False)
    if target_scale is not None:
        scale_factor = target_scale / pointmap_cameras.get_spatial_extent()
        pointmap_cameras = rescale_cameras(pointmap_cameras, scale_factor)
    else:
        scale_factor = 1.
    pm_h, pm_w = scene_pm.points3d.shape[1:3]
    lowres_cameras = CamerasWrapper.from_p3d_cameras(
        p3d_cameras=pointmap_cameras.p3d_cameras,
        height=pm_h,
        width=pm_w,
    )
    matching_thr = matching_thr_factor * pointmap_cameras.get_spatial_extent()
        
    # Build initial depths
    pt_maps = scale_factor * scene_pm.points3d
    imgs = scene_pm.images
    # masks = scene_pm.masks
    manifolds, _ = get_manifold_meshes_from_pointmaps(
        pt_maps, imgs, masks=None, return_single_mesh_object=True, return_manifold_idx=True
    )
    _verts =  torch.nn.Parameter(manifolds.verts_packed().clone().to(device), requires_grad=False).to(device)
    _verts_idx = torch.arange(_verts.shape[0], device=device)

    initial_depths = torch.cat([
        pointmap_cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            _verts.reshape(scene_pm.points3d.shape)[i_chart].reshape(-1, 3)
        )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pointmap_cameras))
    ], dim=0)
    
    # Load matches if needed
    if use_reprojection_loss and reprojection_matches_file is not None:
        match_to_img, match_to_pix, idx_to_image = load_mast3r_matches(reprojection_matches_file)
    else:
        match_to_img, match_to_pix, idx_to_image = None, None, None
    
    pa = ParallelAligner(
        depths=initial_depths,
        cameras=lowres_cameras,
        charts_encoding_params=ChartsEncodingParams(),
        depth_encoding_params=DepthEncodingParams(),
        mlp_params=MLPParams(),
        use_learnable_depth_encoding=use_learnable_depth_encoding,
        learnable_depth_encoding_mode=learnable_depth_encoding_mode,
        use_learnable_confidence=use_learnable_confidence,
        device=device,
        predict_in_disparity_space=predict_in_disparity_space,
        use_meta_mlp=use_meta_mlp,
        use_lora_mlp=use_lora_mlp,
        lora_rank=lora_rank,
        n_lora_layers=n_lora_layers,
        use_multi_res_charts_encoding=use_multi_res_charts_encoding,
        multi_res_charts_encoding_params=MultiResChartsEncodingParams(),
        weight_encodings_with_confidence=weight_encodings_with_confidence,
    )
    
    pa.optimize(
        reference_data=reference_data,
        masks=masks,
        n_iterations=n_iterations,
        use_gradient_loss=use_gradient_loss,
        use_hessian_loss=use_hessian_loss,
        use_normal_loss=use_normal_loss,
        use_matching_loss=use_matching_loss,
        use_curvature_loss=use_curvature_loss,
        use_reprojection_loss=use_reprojection_loss,
        regularize_chart_encodings_norms=regularize_chart_encodings_norms,
        use_total_variation_on_depth_encodings=use_total_variation_on_depth_encodings,
        matching_thr=matching_thr,
        use_confidence_in_matching_loss=use_confidence_in_matching_loss,
        matching_update_iters=matching_update_iters,
        gradient_loss_weight=gradient_loss_weight,
        hessian_loss_weight=hessian_loss_weight,
        normal_loss_weight=normal_loss_weight,
        curvature_loss_weight=curvature_loss_weight,
        matching_loss_weight=matching_loss_weight,
        reprojection_loss_weight=reprojection_loss_weight,
        reprojection_loss_power=reprojection_loss_power,
        chart_encodings_norm_loss_weight=chart_encodings_norm_loss_weight,
        total_variation_on_depth_encodings_weight=total_variation_on_depth_encodings_weight,
        encodings_lr=encodings_lr,
        mlp_lr=mlp_lr,
        confidence_lr=confidence_lr,
        lr_update_iters=lr_update_iters,
        lr_update_factor=lr_update_factor,
        verbose=verbose,
        match_to_img=match_to_img,
        match_to_pix=match_to_pix,
    )
    
    output_verts = pa._deformed_verts.clone()
    output_depths = torch.cat([
    pa.cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
        output_verts[i_chart].reshape(-1, 3)
    )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(pa.cameras))
    ], dim=0)
    
    if use_learnable_confidence:
        with torch.no_grad():
            output_confs = pa.confidence
    else:
        output_confs = 4. * torch.ones_like(output_depths)
    
    if save_charts_data:
        save_path = os.path.join(charts_data_path, "charts_data.npz")
        
        print(f"[INFO] Saving charts data to {save_path}")
        charts_prior_depths = initial_depths
        charts_depths = output_depths
        charts_pts = output_verts
        charts_confs = output_confs
        charts_scale_factor = scale_factor

        # Save the charts data with numpy
        np.savez(
            save_path,
            prior_depths=charts_prior_depths.cpu().numpy(),
            depths=charts_depths.cpu().numpy(),
            pts=charts_pts.cpu().numpy(),
            confs=charts_confs.cpu().numpy(),
            scale_factor=charts_scale_factor,
        )
    
    if use_learnable_confidence:
        if return_training_losses:
            return output_verts, output_depths, output_confs, pa.train_losses
        else:
            return output_verts, output_depths, output_confs
    else:
        if return_training_losses:
            return output_verts, output_depths, pa.train_losses
        else:
            return output_verts, output_depths
        

