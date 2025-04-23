import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import torch
from scene import Scene
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from matcha.dm_extractors.adaptive_tsdf import AdaptiveTSDF
from matcha.dm_utils.rendering import fov2focal
from matcha.dm_scene.cameras import get_cameras_interpolated_between_neighbors


def get_cameras_spatial_extent(views, device):
    from matcha.dm_scene.cameras import GSCamera, CamerasWrapper
    _gs_cameras = []
    for cam in views:
        _gs_cameras.append(
            GSCamera(
                colmap_id=cam.colmap_id, 
                R=cam.R.clone().detach().float().to(device), 
                T=cam.T.clone().detach().float().to(device), # modified as per pytorch warning
                FoVx=cam.FoVx, 
                FoVy=cam.FoVy, 
                image=None, 
                gt_alpha_mask=None,
                image_name=cam.image_name, 
                uid=cam.uid,
                data_device = device,
                image_height=cam.image_height,
                image_width=cam.image_width,
            )
        )
    nerf_cameras = CamerasWrapper(_gs_cameras)
    return nerf_cameras.get_spatial_extent()


@torch.no_grad()
def get_dilated_depth(view, depth, dilation_pixels, rgb=None, max_dilation=1e8):
    """
    Get dilated depth from a depth map.

    Args:
        view (GSCamera): Viewpoint camera.
        depth (torch.Tensor): Depth map. Should have shape (H, W) or (1, H, W), or (H, W, 1).
        dilation_factor (float): Dilation factor.
        rgb (torch.Tensor, optional): RGB image. Defaults to None. Has shape (3, H, W) or (H, W, 3).
    """
    from matcha.dm_scene.charts import depths_to_points_parallel
    from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps, Meshes, TexturesVertex, render_mesh_with_pytorch3d
    from matcha.dm_scene.cameras import GSCamera, CamerasWrapper
    
    # Get camera object
    gs_cameras = []
    for cam in [view]:
        gs_cameras.append(
            GSCamera(
                colmap_id=cam.colmap_id, 
                R=torch.tensor(cam.R).float().to(depth.device), 
                T=torch.tensor(cam.T).float().to(depth.device), 
                FoVx=cam.FoVx, 
                FoVy=cam.FoVy, 
                image=None, 
                gt_alpha_mask=None,
                image_name=cam.image_name, 
                uid=cam.uid,
                data_device = "cuda",
                image_height=cam.image_height,
                image_width=cam.image_width,
            )
        )
    nerf_cameras = CamerasWrapper(gs_cameras)
    
    # Get 3D points from depth map
    depth_view = depth.squeeze()  # (H, W)
    if False:
        # Depth to points function from 2DGS leads to imprecision in the backprojection.
        pts = depths_to_points_parallel(depth_view[None], [view]).view(*depth_view.shape, 3)  # (H, W, 3)
    else:
        pts = nerf_cameras.backproject_depth(cam_idx=0, depth=depth_view).view(*depth_view.shape, 3)
    rgb_channels_first = False
    if rgb is not None:
        if rgb.shape[0] == 3:
            rgb_view = rgb.permute(1, 2, 0)  # (H, W, 3)
            rgb_channels_first = True
        else:
            rgb_view = rgb  # (H, W, 3)
    else:
        rgb_view = torch.ones_like(pts)
    
    # Build surface mesh
    meshes = get_manifold_meshes_from_pointmaps(
        points3d=pts[None],  # (1, H, W, 3)
        imgs=rgb_view[None],  # (1, H, W, 3)
        return_single_mesh_object=False
    )
    verts, faces = meshes[0].verts_packed(), meshes[0].faces_packed()
    verts_normals = meshes[0].verts_normals_packed()
    verts_features = meshes[0].textures.verts_features_packed()
    
    # Compute dilation factor, depending on the depth (dilate more in distant areas)
    focals = (nerf_cameras.fx[0] + nerf_cameras.fy[0]).item() / 2  # focal length in pixels
    dilation_factor = dilation_pixels / focals * depth.view(-1, 1)
    dilation_factor = dilation_factor.clamp(max=max_dilation)
    
    # Dilate surface mesh along normals
    dilated_verts = verts + dilation_factor * verts_normals
    dilated_mesh = Meshes(
        verts=[dilated_verts], 
        faces=[faces],
        textures=TexturesVertex(verts_features=[verts_features])
    )
    
    # Render dilated mesh to get dilated depth and rgb
    render_pkg = render_mesh_with_pytorch3d(
        dilated_mesh, 
        nerf_cameras=nerf_cameras, 
        camera_idx=0,
    )
    dilated_rgb = render_pkg["rgb"]
    dilated_depth = render_pkg["depth"]
    dilated_depth[dilated_depth == 0.] = depth_view[dilated_depth == 0.]
    dilated_rgb[dilated_depth == 0.] = rgb_view[dilated_depth == 0.]
    if rgb_channels_first:
        dilated_rgb = dilated_rgb.permute(2, 0, 1)
    return dilated_depth.view(*depth.shape), dilated_rgb.view(*rgb.shape)


@torch.no_grad()
def evaluate_tsdf(points, views, gaussians:GaussianModel, pipeline, background, return_color=False, 
                  truncation_margin=0.005, interpolate_depth=True, interpolation_mode='bilinear', weight_interpolation_by_depth_gradient=False,
                  use_dilated_depth=False, use_sdf_tolerance=False, use_unbiased_tsdf=False,
                  use_binary_opacity=False, filter_with_depth_gradient=False, filter_with_normal_consistency=False, 
                  weight_by_softmax=False, weight_by_normal_consistency=False,
                  softmax_temperature=1.0,
                ):
    spatial_extent = get_cameras_spatial_extent(views, device=points.device)
    
    # Initialize adaptive TSDF
    adaptive_tsdf = AdaptiveTSDF(
        points=points.view(-1, 3),
        trunc_margin=truncation_margin * spatial_extent,
        znear=1e-6,
        zfar=1e6,
        use_binary_opacity=use_binary_opacity,
    )
    
    # Integrate depth maps
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            render_pkg = render(view, gaussians, pipeline, background)
            depth = render_pkg['surf_depth']
            rgb = render_pkg["render"]
            normals = render_pkg["surf_normal"]
            rendered_normals = render_pkg["rend_normal"]
            
            if use_dilated_depth:
                dilation_pixels = 1.5  # 3., 2., 1.5
                max_dilation = 1e-3 * spatial_extent  # 0.01
                depth, rgb = get_dilated_depth(
                    view, depth, dilation_pixels=dilation_pixels, rgb=rgb, max_dilation=max_dilation
                )
                
            if use_sdf_tolerance:
                tolerance_in_pixels = 1.5  # 3., 2., 1.5
                max_tolerance = 1e-3 * spatial_extent  # 0.01
                focals = (
                    fov2focal(view.FoVx, view.image_width) 
                    + fov2focal(view.FoVy, view.image_height)
                ) / 2  # focal length in pixels
                tolerance = (tolerance_in_pixels / focals * depth).clamp(max=max_tolerance)
                depth = depth - tolerance
            
            adaptive_tsdf.integrate(
                img=rgb,
                depth=depth,
                camera=view,
                obs_weight=1.0,
                interpolate_depth=interpolate_depth,
                interpolation_mode=interpolation_mode,
                weight_interpolation_by_depth_gradient=weight_interpolation_by_depth_gradient,
                depth_gradient_threshold=0.2 * spatial_extent,
                normals=normals if (use_unbiased_tsdf or filter_with_normal_consistency or weight_by_normal_consistency) else None,
                filter_with_depth_gradient=filter_with_depth_gradient,
                depth_gradient_threshold_for_filtering=0.1 * spatial_extent,
                unbias_depth_using_normals=use_unbiased_tsdf,
                filter_with_normal_consistency=filter_with_normal_consistency,
                normal_consistency_threshold=0.5,
                reference_normals=rendered_normals if (filter_with_normal_consistency or weight_by_normal_consistency) else None,
                weight_by_softmax=weight_by_softmax,
                softmax_temperature=softmax_temperature,
                weight_by_normal_consistency=weight_by_normal_consistency,
            )
    
    # Get final field values
    tsdf_pkg = adaptive_tsdf.return_field_values()
    tsdf = tsdf_pkg["tsdf"].view(-1)
    colors = tsdf_pkg["colors"].view(-1, 3)
    
    print(f"Number of points with tsdf < 0: {len(tsdf[tsdf < 0])} / {len(tsdf)}")
    print(f"Number of points with tsdf > 0: {len(tsdf[tsdf > 0])} / {len(tsdf)}")
    
    if return_color:
        return tsdf, colors
    return tsdf


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    model_path : str, 
    name : str, 
    iteration : int, 
    views, 
    gaussians : GaussianModel, 
    pipeline, 
    background : torch.Tensor, 
    filter_mesh : bool, 
    texture_mesh : bool, 
    downsample_ratio : float,
    gaussian_flatness : float,
    output_dir : str = None,
    truncation_margin : float = 0.001,
    interpolate_depth : bool = True,
    interpolation_mode : str = 'bilinear',
    weight_interpolation_by_depth_gradient : bool = False,
    use_dilated_depth : bool = False,
    use_sdf_tolerance : bool = False,
    use_unbiased_tsdf : bool = False,
    use_binary_opacity : bool = False,
    filter_with_depth_gradient : bool = False,
    filter_with_normal_consistency : bool = False,
    weight_by_softmax : bool = False,
    weight_by_normal_consistency : bool = False,
    softmax_temperature : float = 1.0,
    save_cells : bool = False,
):
    if output_dir is None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")
    else:
        render_path = output_dir
    makedirs(render_path, exist_ok=True)
    
    # Compute spatial extent
    spatial_extent = get_cameras_spatial_extent(views, device='cuda')
    
    # generate tetra points here
    # TODO: Multiply gaussian_flatness by spatial_extent
    points, points_scale = gaussians.get_tetra_points(downsample_ratio=downsample_ratio, gaussian_flatness=gaussian_flatness * spatial_extent)
    # load cell if exists
    cells_file_path = os.path.join(render_path, "cells.pt")
    if os.path.exists(cells_file_path):
        print("load existing cells")
        cells = torch.load(cells_file_path)
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
        # we should filter the cell if it is larger than the gaussians
        if save_cells:
            torch.save(cells, cells_file_path)
    
    # evaluate SDF
    if interpolate_depth:
        print("[INFO] Interpolating depth values for TSDF integration.")
        if weight_interpolation_by_depth_gradient:
            print("[INFO]    > Weighting interpolation by depth gradient.")
    if use_dilated_depth:
        print("[INFO] Using dilated depth for TSDF integration.")
    if use_sdf_tolerance:
        print("[INFO] Using SDF tolerance for TSDF integration.")
    if use_unbiased_tsdf:
        print("[INFO] Using unbiased TSDF for TSDF integration.")
    if use_binary_opacity:
        print("[INFO] Using binary opacity for TSDF integration.")
    if filter_with_depth_gradient:
        print("[INFO] Filtering with depth gradient.")
    if filter_with_normal_consistency:
        print("[INFO] Filtering with normal consistency.")
    if weight_by_softmax:
        print(f"[INFO] Weighting by softmax with temperature {softmax_temperature}.")
    if weight_by_normal_consistency:
        print("[INFO] Weighting by normal consistency.")
    print(f"[INFO] Rendering depth maps with depth ratio: {pipeline.depth_ratio}")
    sdf = evaluate_tsdf(points, views, gaussians, pipeline, background, truncation_margin=truncation_margin,
                        interpolate_depth=interpolate_depth, interpolation_mode=interpolation_mode, 
                        weight_interpolation_by_depth_gradient=weight_interpolation_by_depth_gradient, use_dilated_depth=use_dilated_depth,
                        use_sdf_tolerance=use_sdf_tolerance, use_unbiased_tsdf=use_unbiased_tsdf, use_binary_opacity=use_binary_opacity,
                        filter_with_depth_gradient=filter_with_depth_gradient, filter_with_normal_consistency=filter_with_normal_consistency,
                        weight_by_softmax=weight_by_softmax, weight_by_normal_consistency=weight_by_normal_consistency, 
                        softmax_temperature=softmax_temperature)
    sdf = sdf[None]

    vertices = points.cuda()[None]
    tets = cells.cuda().long()
    
    print(vertices.shape, tets.shape, sdf.shape)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        mid_sdf = evaluate_tsdf(mid_points, views, gaussians, pipeline, background, truncation_margin=truncation_margin,
                                interpolate_depth=interpolate_depth, interpolation_mode=interpolation_mode, 
                                weight_interpolation_by_depth_gradient=weight_interpolation_by_depth_gradient, use_dilated_depth=use_dilated_depth,
                                use_sdf_tolerance=use_sdf_tolerance, use_unbiased_tsdf=use_unbiased_tsdf, use_binary_opacity=use_binary_opacity,
                                filter_with_depth_gradient=filter_with_depth_gradient, filter_with_normal_consistency=filter_with_normal_consistency,
                                weight_by_softmax=weight_by_softmax, weight_by_normal_consistency=weight_by_normal_consistency,
                                softmax_temperature=softmax_temperature)
        mid_sdf = mid_sdf[None]
        mid_sdf = mid_sdf.squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        if step not in [n_binary_steps - 1]:
            continue
        
        if texture_mesh:
            active_sh_degree = gaussians.active_sh_degree
            gaussians.active_sh_degree = 0
            _, color = evaluate_tsdf(points, views, gaussians, pipeline, background, truncation_margin=truncation_margin, return_color=True,
                                     interpolate_depth=interpolate_depth, interpolation_mode=interpolation_mode, 
                                     weight_interpolation_by_depth_gradient=weight_interpolation_by_depth_gradient, use_dilated_depth=use_dilated_depth,
                                     use_sdf_tolerance=use_sdf_tolerance, use_unbiased_tsdf=use_unbiased_tsdf, use_binary_opacity=use_binary_opacity,
                                     filter_with_depth_gradient=filter_with_depth_gradient, filter_with_normal_consistency=filter_with_normal_consistency,
                                     weight_by_softmax=weight_by_softmax, weight_by_normal_consistency=weight_by_normal_consistency,
                                     softmax_temperature=softmax_temperature)
            vertex_colors=(color.clamp(min=0., max=1.).cpu().numpy() * 255).astype(np.uint8)
            gaussians.active_sh_degree = active_sh_degree
        else:
            vertex_colors=None
        mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
        # filter
        if filter_mesh:
            mask = (distance <= scale).cpu().numpy()
            face_mask = mask[faces].all(axis=1)
            mesh.update_vertices(mask)
            mesh.update_faces(face_mask)
        
        mesh.export(os.path.join(render_path, f"tetra_mesh_binary_search_{step}.ply"))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))
    

def extract_mesh(
    dataset : ModelParams, 
    iteration : int, 
    pipeline : PipelineParams, 
    filter_mesh : bool, 
    texture_mesh : bool, 
    downsample_ratio : float,
    gaussian_flatness : float,
    output_dir : str = None,
    truncation_margin : float = 0.005,
    interpolate_depth : bool = True,
    interpolation_mode : str = 'bilinear',
    weight_interpolation_by_depth_gradient : bool = False,
    use_dilated_depth : bool = False,
    use_sdf_tolerance : bool = False,
    use_unbiased_tsdf : bool = False,
    use_binary_opacity : bool = False,
    filter_with_depth_gradient : bool = False,
    filter_with_normal_consistency : bool = False,
    weight_by_softmax : bool = False,
    weight_by_normal_consistency : bool = False,
    softmax_temperature : float = 1.0,
    interpolate_cameras : bool = False,
    n_neighbors_to_interpolate : int = 2,
    n_interpolated_cameras_for_each_neighbor : int = 10,
    dense_data_path : str = None,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        print(("Dataset: ", dataset.source_path, dataset.model_path))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Dense data
        if dense_data_path is not None:
            print(f"[INFO] Loading dense supervision data from: {dense_data_path}")
            dense_dataset = copy.deepcopy(dataset)
            dense_dataset.source_path = dense_data_path
            dense_dataset.model_path = os.path.join(dataset.model_path, 'dense_data')
            dense_gaussians = GaussianModel(dataset.sh_degree)
            dense_scene = Scene(dense_dataset, dense_gaussians, shuffle=False)
            _cams = dense_scene.getTrainCameras()
            print(f"          > Number of dense cameras: {len(_cams)}")
            print(f"          > Pseudo-views interpolation will be disabled because dense supervision is provided.")
            interpolate_cameras = False
        else:
            _cams = scene.getTrainCameras()
        
        if interpolate_cameras:
            print(f"[INFO] Pseudo-views interpolated between training views will be used for TSDF integration.")
            print(f"          > Interpolating between {n_neighbors_to_interpolate} neighbors for each camera.")
            print(f"          > Interpolating {n_interpolated_cameras_for_each_neighbor} views for each neighbor.")
            cams = get_cameras_interpolated_between_neighbors(
                cameras=_cams, 
                n_neighbors_to_interpolate=n_neighbors_to_interpolate, 
                n_interpolated_cameras_for_each_neighbor=n_interpolated_cameras_for_each_neighbor
            )
        else:
            cams = _cams
        
        marching_tetrahedra_with_binary_search(
            model_path=dataset.model_path, 
            name="test", 
            iteration=iteration, 
            views=cams, 
            gaussians=gaussians, 
            pipeline=pipeline, 
            background=background, 
            filter_mesh=filter_mesh, 
            texture_mesh=texture_mesh, 
            downsample_ratio=downsample_ratio,
            gaussian_flatness=gaussian_flatness,
            output_dir=output_dir,
            truncation_margin=truncation_margin,
            interpolate_depth=interpolate_depth,
            interpolation_mode=interpolation_mode,
            weight_interpolation_by_depth_gradient=weight_interpolation_by_depth_gradient,
            use_dilated_depth=use_dilated_depth,
            use_sdf_tolerance=use_sdf_tolerance,
            use_unbiased_tsdf=use_unbiased_tsdf,
            use_binary_opacity=use_binary_opacity,
            filter_with_depth_gradient=filter_with_depth_gradient,
            filter_with_normal_consistency=filter_with_normal_consistency,
            weight_by_softmax=weight_by_softmax,
            weight_by_normal_consistency=weight_by_normal_consistency,
            softmax_temperature=softmax_temperature,
            save_cells=False,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--downsample_ratio", type=float, default=None)
    parser.add_argument("--gaussian_flatness", type=float, default=2e-4)  # 1e-3 / spatial_extent = 2e-4 works well
    parser.add_argument("--truncation_margin", type=float, default=5e-3)  # 5. * 1e-3 = 25. * gaussian_flatness works well
    parser.add_argument("--interpolate_depth", action="store_true")
    parser.add_argument("--interpolation_mode", type=str, default='bilinear')
    parser.add_argument("--weight_interpolation_by_depth_gradient", action="store_true")
    parser.add_argument("--use_dilated_depth", action="store_true")
    parser.add_argument("--use_sdf_tolerance", action="store_true")
    parser.add_argument("--use_unbiased_tsdf", action="store_true")
    parser.add_argument("--use_binary_opacity", action="store_true")
    parser.add_argument("--filter_with_depth_gradient", action="store_true")
    parser.add_argument("--filter_with_normal_consistency", action="store_true")
    parser.add_argument("--weight_by_softmax", action="store_true")
    parser.add_argument("--weight_by_normal_consistency", action="store_true")
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    # Interpolate cameras
    parser.add_argument("--interpolate_cameras", action="store_true")
    parser.add_argument("--n_neighbors_to_interpolate", type=int, default=2)
    parser.add_argument("--n_interpolated_cameras_for_each_neighbor", type=int, default=10)
    # Dense supervision (Optional)
    parser.add_argument("--dense_data_path", type=str, default="none")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # args.truncation_margin = args.truncation_margin * args.gaussian_flatness

    if args.dense_data_path == "none":
        print("[INFO] Dense supervision is not provided.")
        args.dense_data_path = None

    extract_mesh(
        dataset=model.extract(args), 
        iteration=args.iteration, 
        pipeline=pipeline.extract(args), 
        filter_mesh=args.filter_mesh, 
        texture_mesh=args.texture_mesh, 
        downsample_ratio=args.downsample_ratio,
        gaussian_flatness=args.gaussian_flatness,
        output_dir=args.output_dir,
        truncation_margin=args.truncation_margin,
        interpolate_depth=args.interpolate_depth,
        interpolation_mode=args.interpolation_mode,
        weight_interpolation_by_depth_gradient=args.weight_interpolation_by_depth_gradient,
        use_dilated_depth=args.use_dilated_depth,
        use_sdf_tolerance=args.use_sdf_tolerance,
        use_unbiased_tsdf=args.use_unbiased_tsdf,
        use_binary_opacity=args.use_binary_opacity,
        filter_with_depth_gradient=args.filter_with_depth_gradient,
        filter_with_normal_consistency=args.filter_with_normal_consistency,
        weight_by_softmax=args.weight_by_softmax,
        weight_by_normal_consistency=args.weight_by_normal_consistency,
        softmax_temperature=args.softmax_temperature,
        interpolate_cameras=args.interpolate_cameras,
        n_neighbors_to_interpolate=args.n_neighbors_to_interpolate,
        n_interpolated_cameras_for_each_neighbor=args.n_interpolated_cameras_for_each_neighbor,
        dense_data_path=args.dense_data_path,
    )
