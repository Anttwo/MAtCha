from typing import List
import numpy as np
import torch
from torch.nn.functional import normalize as torch_normalize
import open3d as o3d

from matcha.dm_scene.cameras import CamerasWrapper
from matcha.dm_scene.learnable_cameras import LearnableCameras
from matcha.dm_scene.meshes import get_manifold_meshes_from_pointmaps, remove_faces_from_single_mesh
from matcha.dm_modules.adaln import AdaLN, initialize_adaln_weights
from matcha.dm_modules.matcher_3d import Matcher3D, get_points_depth_in_depthmap_parallel
from matcha.dm_deformation.encodings import ChartsEncoding, DepthEncoding, MultiResChartsEncoding
from matcha.dm_deformation.multi_mlp import DeformationMultiMLP, initialize_multi_mlp_weights
from matcha.dm_deformation.meta_multi_mlp import DeformationMetaMultiMLP, initialize_meta_multi_mlp_weights
from matcha.dm_deformation.lora_multi_mlp import DeformationLoRAMultiMLP, initialize_lora_multi_mlp_weights
from matcha.dm_utils.image import img_grad, img_hessian
from matcha.dm_utils.rendering import(
    depths_to_points_parallel,
    depth2normal_parallel,
    normal2curv_parallel,
)
from matcha.pointmap.depthanythingv2 import get_points_depth_in_depthmap
from matcha.pointmap.mast3rcontroller import get_minimal_projections_diffs
from tqdm import tqdm


# ----- ParallelAligner parameters -----

class ChartsEncodingParams():
    def __init__(
        self,
        encoding_dim:int=32,  # TODO: Changed
        resolution_factor:float=0.4,  # TODO: Changed
        initialization_range:float=1e-4,
    ):
        self.encoding_dim = encoding_dim
        self.resolution_factor = resolution_factor
        self.initialization_range = initialization_range
        

class MultiResChartsEncodingParams():
    def __init__(
        self,
        encoding_dim_per_res:int=8,
        resolutions:List[int]=[0.05, 0.1, 0.2, 0.4],
        initialization_range:float=1e-4,
    ):
        self.encoding_dim_per_res = encoding_dim_per_res
        self.encoding_dim = encoding_dim_per_res * len(resolutions)
        self.resolutions = resolutions
        self.initialization_range = initialization_range


class DepthEncodingParams():
    def __init__(
        self,
        encoding_dim:int=32,  # TODO: Changed
        n_bins:int=30,  # TODO: Changed (30)
        initialization_range:float=1e-4,
    ):
        self.encoding_dim = encoding_dim
        self.n_bins = n_bins
        self.initialization_range = initialization_range


class MLPParams():
    def __init__(
        self,
        n_deformation_layers:int=3,  # TODO: Changed
        deformation_layer_size:int=64,  # TODO: Changed
        scene_radius_factor:float=2.,
        deformation_radius_factor:float=1.,
        no_final_linearity:bool=True,
    ):
        self.n_deformation_layers = n_deformation_layers
        self.deformation_layer_size = deformation_layer_size
        self.scene_radius_factor = scene_radius_factor
        self.deformation_radius_factor = deformation_radius_factor
        self.no_final_linearity = no_final_linearity
        
        
# Old confidence to increasing weight function
if True:
    def confidence_to_weight(confidence:torch.Tensor):
        conf_weights = confidence - 1.
        return 1. - torch.exp(-conf_weights**2 / 2)
# New confidence to increasing weight function
else:
    def confidence_to_weight(confidence:torch.Tensor):
        conf_weights = confidence - 1.
        return torch.sigmoid((conf_weights - 2.) * 2.)


# ----- ParallelAligner -----
        
class ParallelAligner(torch.nn.Module):
    """Class to align depth maps/charts to other depth maps/charts or observed point clouds, while preserving the initial structure.
    """
    def __init__(
        self, 
        depths:torch.Tensor,
        cameras:CamerasWrapper,
        charts_encoding_params:ChartsEncodingParams=ChartsEncodingParams(),
        depth_encoding_params:DepthEncodingParams=DepthEncodingParams(),
        mlp_params:MLPParams=MLPParams(),
        use_learnable_depth_encoding:bool=False,
        learnable_depth_encoding_mode:str='add',
        device='cuda',
        predict_in_disparity_space:bool=False,
        use_learnable_confidence:bool=True,
        use_meta_mlp:bool=False,
        use_lora_mlp:bool=False,
        lora_rank:int=4,
        n_lora_layers:int=2,
        use_multi_res_charts_encoding:bool=False,
        multi_res_charts_encoding_params:MultiResChartsEncodingParams=MultiResChartsEncodingParams(),
        weight_encodings_with_confidence:bool=False,
        learnable_extrinsics:bool=False,
        learnable_intrinsics:bool=False,
    ) -> None:
        """
        Args:
            depths (torch.Tensor): Has shape (n_charts, h, w).
            cameras (CamerasWrapper): Should contain n_charts cameras.
            charts_encoding_params (ChartsEncodingParams): _description_
            depth_encoding_params (DepthEncodingParams): _description_
            mlp_params (MLPParams): _description_
            use_learnable_depth_encoding (bool, optional): _description_. Defaults to False.
        """
        super(ParallelAligner, self).__init__()
        
        assert depths.shape[0] == len(cameras)
        
        self.use_learnable_depth_encoding = use_learnable_depth_encoding
        self.learnable_depth_encoding_mode = learnable_depth_encoding_mode
        
        n_pm, pm_h, pm_w = depths.shape[:3]
        self.n_pm, self.pm_h, self.pm_w = n_pm, pm_h, pm_w
        chart_h, chart_w = int(charts_encoding_params.resolution_factor * pm_h), int(charts_encoding_params.resolution_factor * pm_w)
        self.chart_h, self.chart_w = chart_h, chart_w
        no_final_linearity = mlp_params.no_final_linearity
        predict_deformations_along_rays = True
        self.predict_deformations_along_rays = predict_deformations_along_rays
        self.predict_in_disparity_space = predict_in_disparity_space
        self.charts_encoding_params = charts_encoding_params
        self.depth_encoding_params = depth_encoding_params
        self.mlp_params = mlp_params
        self.use_meta_mlp = use_meta_mlp
        self.use_lora_mlp = use_lora_mlp
        
        self.weight_encodings_with_confidence = weight_encodings_with_confidence
        
        self.learnable_extrinsics = learnable_extrinsics
        self.learnable_intrinsics = learnable_intrinsics
        
        if (self.use_meta_mlp or self.use_lora_mlp) and not self.use_learnable_depth_encoding:
            raise ValueError("Meta MLP or LoRA MLP can only be used with learnable depth encoding.")
        if self.use_lora_mlp and self.use_meta_mlp:
            raise ValueError("LoRA MLP and Meta MLP cannot be used together.")
        if self.weight_encodings_with_confidence and not use_learnable_confidence:
            raise ValueError("Encodings cannot be weighted with confidence if confidence is not learnable.")
        
        if weight_encodings_with_confidence:
            print("[INFO] Encodings will be weighted with confidence.")
        
        self.cameras = cameras
        
        # Create 3D points and rays
        _verts = torch.zeros(n_pm, pm_h, pm_w, 3, device=device)
        for i_depth in range(n_pm):
            _verts[i_depth] = cameras.backproject_depth(cam_idx=i_depth, depth=depths[i_depth]).view(pm_h, pm_w, 3)
        self._verts = torch.nn.Parameter(_verts, requires_grad=False).to(device)
        self._depths = torch.nn.Parameter(depths, requires_grad=False).to(device)
        self._rays = (_verts - cameras.p3d_cameras.get_camera_center().view(n_pm, 1, 1, 3)).view(n_pm, -1 , 3)  # (n_charts, n_verts_per_chart, 3)
        self._deformed_verts = torch.nn.Parameter(_verts.clone(), requires_grad=False).to(device)
        
        # Create chart encoding
        self.use_multi_res_charts_encoding = use_multi_res_charts_encoding
        if use_multi_res_charts_encoding:
            self.charts_encoding_params = multi_res_charts_encoding_params
            self.charts_encoding = MultiResChartsEncoding(
                num_charts=n_pm,
                height=pm_h,
                width=pm_w,
                resolutions=multi_res_charts_encoding_params.resolutions,
                encoding_dim_per_res=multi_res_charts_encoding_params.encoding_dim_per_res,
                initialization_range=multi_res_charts_encoding_params.initialization_range,
            ).to(device)
        else:
            self.charts_encoding = ChartsEncoding(
                num_charts=n_pm,
                encoding_h=chart_h,
                encoding_w=chart_w,
                encoding_dim=charts_encoding_params.encoding_dim,
                initialization_range=charts_encoding_params.initialization_range,
            ).to(device)
        self._pts_uv = torch.nn.Parameter(
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1., 1., pm_h, device=device),
                    torch.linspace(-1., 1., pm_w, device=device)
                ), dim=-1
            ).repeat(n_pm, 1, 1, 1), 
            requires_grad=False,
        ).to(device)

        # Create MLP
        scene_radius = mlp_params.scene_radius_factor * cameras.get_spatial_extent()
        deformation_radius = mlp_params.deformation_radius_factor * cameras.get_spatial_extent()
        _additional_input_dim = (
            charts_encoding_params.encoding_dim if (use_learnable_depth_encoding and learnable_depth_encoding_mode=='concatenate')
            else 0)
        if self.use_meta_mlp:
            self.deformation = DeformationMetaMultiMLP(
                n_heads=n_pm,
                n_layer=mlp_params.n_deformation_layers,
                layer_size=mlp_params.deformation_layer_size,
                input_dim=charts_encoding_params.encoding_dim,
                cond_dim=depth_encoding_params.encoding_dim,
                output_dim=1 if predict_deformations_along_rays else 3,
                additional_input_dim=_additional_input_dim,
                data_input_range_min=-scene_radius,
                data_input_range_max=scene_radius,
                mlp_input_range_min=-1.,
                mlp_input_range_max=1.,
                output_range_min=-deformation_radius,
                output_range_max=deformation_radius,
                non_linearity=torch.nn.ReLU(),
                final_non_linearity=None if no_final_linearity else torch.nn.Sigmoid(),
                positional_encoding=None,
            )
            initialize_meta_multi_mlp_weights(self.deformation, std=None)
        elif self.use_lora_mlp:
            self.deformation = DeformationLoRAMultiMLP(
                n_heads=n_pm,
                n_layer=mlp_params.n_deformation_layers,
                layer_size=mlp_params.deformation_layer_size,
                input_dim=charts_encoding_params.encoding_dim,
                output_dim=1 if predict_deformations_along_rays else 3,
                cond_dim=depth_encoding_params.encoding_dim,
                lora_rank=lora_rank,
                n_lora_layers=n_lora_layers,
                additional_input_dim=_additional_input_dim,
                data_input_range_min=-scene_radius,
                data_input_range_max=scene_radius,
                mlp_input_range_min=-1.,
                mlp_input_range_max=1.,
                output_range_min=-deformation_radius,
                output_range_max=deformation_radius,
                non_linearity=torch.nn.ReLU(),
                final_non_linearity=None if no_final_linearity else torch.nn.Sigmoid(),
                positional_encoding=None,
            )
            initialize_lora_multi_mlp_weights(self.deformation, std=None)
        else:
            self.deformation = DeformationMultiMLP(
                n_heads=n_pm,
                n_layer=mlp_params.n_deformation_layers,
                layer_size=mlp_params.deformation_layer_size,
                input_dim=charts_encoding_params.encoding_dim,
                output_dim=1 if predict_deformations_along_rays else 3,
                additional_input_dim=_additional_input_dim,
                data_input_range_min=-scene_radius,
                data_input_range_max=scene_radius,
                mlp_input_range_min=-1.,
                mlp_input_range_max=1.,
                output_range_min=-deformation_radius,
                output_range_max=deformation_radius,
                non_linearity=torch.nn.ReLU(),
                final_non_linearity=None if no_final_linearity else torch.nn.Sigmoid(),
                positional_encoding=None,
            ).to(device)
            # Initialize MLP with Xavier initialization
            initialize_multi_mlp_weights(self.deformation, std=None)

        # Create Depth Encoding
        # 1. Create depth coordinates for all vertices in each chart
        depth_coords = []
        for _i_depth in range(n_pm):
            # Compute depth of each vertex in the chart
            depth_i = self._depths[_i_depth].view(-1)
            
            # Compute quantiles of the depth
            n_quantiles_to_use_for_learnable_distance_encoding = depth_encoding_params.n_bins
            _quantiles_i = torch.zeros(n_quantiles_to_use_for_learnable_distance_encoding, device=device, dtype=torch.float32)
            _depth_bins_i = torch.zeros(depth_i.shape[0], device=device, dtype=torch.int32)
            _depth_coords_i = torch.zeros(depth_i.shape[0], device=device, dtype=torch.float32)
            for i_quantile in range(n_quantiles_to_use_for_learnable_distance_encoding):
                _quantiles_i[i_quantile] = torch.quantile(depth_i, i_quantile / n_quantiles_to_use_for_learnable_distance_encoding).item()
            
            # Compute depth bins
            for i_quantile in range(n_quantiles_to_use_for_learnable_distance_encoding):
                quantile_mask = (
                    (depth_i >= _quantiles_i[i_quantile]) & (depth_i < _quantiles_i[i_quantile + 1])
                    ) if i_quantile < n_quantiles_to_use_for_learnable_distance_encoding - 1 else (
                        depth_i >= _quantiles_i[i_quantile]
                        )
                _depth_bins_i[quantile_mask] = i_quantile
                if i_quantile < n_quantiles_to_use_for_learnable_distance_encoding - 1:
                    _depth_coords_i[quantile_mask] = (depth_i[quantile_mask] - _quantiles_i[i_quantile]) / (_quantiles_i[i_quantile + 1] - _quantiles_i[i_quantile]) + i_quantile
                    _depth_coords_i[quantile_mask] = _depth_coords_i[quantile_mask] / (n_quantiles_to_use_for_learnable_distance_encoding - 1)
                else:
                    _depth_coords_i[quantile_mask] = 1.
            depth_coords.append(_depth_coords_i[None])
        depth_coords = torch.cat(depth_coords, dim=0)
        self.depth_coords = torch.nn.Parameter(depth_coords, requires_grad=False).to(device)  # (n_charts, n_verts_per_chart)

        # Create depth encodings
        self.depth_encoding = DepthEncoding(
            num_charts=n_pm,
            num_bins=n_quantiles_to_use_for_learnable_distance_encoding,
            encoding_dim=depth_encoding_params.encoding_dim,
            initialization_range=depth_encoding_params.initialization_range,
        ).to(device)
        
        if use_learnable_depth_encoding and learnable_depth_encoding_mode == 'adaln':
            self.adaln = AdaLN(dim=depth_encoding_params.encoding_dim).to(device)
            initialize_adaln_weights(self.adaln, std=None)
            
        # Create confidence if needed
        self.use_learnable_confidence = use_learnable_confidence
        if use_learnable_confidence:
            print("""
                  [WARNING][TODO] Confidence should be instantiated in self.prepare_for_optimization as
                  the number of parameters depends on the format of the reference data.
                  If the reference data is a list of point clouds with different number of points,
                  then the confidence should be a list of optimizable parameters.
                  """
                )
            self._confidence = torch.nn.Parameter(
                torch.zeros(n_pm, pm_h, pm_w, device=device), 
                requires_grad=True
            ).to(device)
            self.confidence_weighting = 0.2
            
        if learnable_extrinsics or learnable_intrinsics:
            print("[INFO] Extrinsics and intrinsics are learnable.")
            self.cameras = LearnableCameras.from_gs_cameras(
                gs_cameras=self.cameras.gs_cameras, 
                learnable_intrinsics=learnable_intrinsics, 
                learnable_extrinsics=learnable_extrinsics
            )
    
    @property
    def device(self):
        return self._verts.device
    
    @property
    def verts_deformations(self):
        # Compute encodings
        _charts_encoding_dim = self.charts_encoding_params.encoding_dim        
        encodings = self.charts_encoding(self._pts_uv).reshape(self.n_pm, -1, _charts_encoding_dim)  # (n_depth, n_verts_per_chart, charts_encoding_dim)
        
        # Weight encodings with confidence if needed
        if self.weight_encodings_with_confidence:
            if True:
                conf_weights = (self.confidence.detach() - 1.).view(self.n_pm, -1, 1)  # (n_depth, n_verts_per_chart, 1)
                conf_weights = 1. - torch.exp(-conf_weights**2 / 2)  # TODO: Changed
            else:
                conf_weights = confidence_to_weight(self.confidence.detach().view(self.n_pm, -1, 1))
                
            encodings = encodings * conf_weights
            
        # Compute and add depth encoding if needed
        additional_input = None
        if self.use_learnable_depth_encoding:            
            depth_encodings = self.depth_encoding(self.depth_coords).reshape(self.n_pm, -1, _charts_encoding_dim)  # (n_depth, n_verts_per_chart, charts_encoding_dim)
            if not (self.use_meta_mlp or self.use_lora_mlp):
                if self.learnable_depth_encoding_mode == 'add':
                    encodings = encodings + depth_encodings
                elif self.learnable_depth_encoding_mode == 'multiply':
                    encodings = encodings * depth_encodings
                elif self.learnable_depth_encoding_mode == 'replace':
                    encodings = depth_encodings
                elif self.learnable_depth_encoding_mode == 'concatenate':
                    additional_input = depth_encodings
                elif self.learnable_depth_encoding_mode == 'adaln':
                    encodings = encodings + 0.001 * self.adaln(encodings, depth_encodings)
                else:
                    raise ValueError(f"learnable_depth_encoding_mode must be either 'add', 'replace', \
                                        or 'concatenate', and not {self.learnable_depth_encoding_mode}.")
            
        # Compute deformations
        if self.use_meta_mlp or self.use_lora_mlp:
            deformations = self.deformation(encodings, cond=depth_encodings)
        else:
            deformations = self.deformation(encodings, additional_input=additional_input)  # (n_depth, n_verts_per_chart, 3 or 1)
        
        # If we are predicting deformations along rays, we need to multiply the deformations by the ray directions
        # Moreover, if we are predicting in disparity space, we need to scale the deformations by the distance to the camera center
        predict_deformations_along_rays = self.deformation.output_dim == 1
        if predict_deformations_along_rays or self.predict_in_disparity_space:
            if self._rays is None:
                raise ValueError("Rays must be provided if we are predicting deformations along rays or in disparity space.")
        if predict_deformations_along_rays:
            if self.predict_in_disparity_space:
                mlp_output_scale = (self.deformation.output_range_max - self.deformation.output_range_min) / 2
                deformations = deformations / mlp_output_scale * self._rays
            else:
                deformations = deformations * torch_normalize(self._rays, dim=-1)
        else:
            if self.predict_in_disparity_space:
                mlp_output_scale = (self.deformation.output_range_max - self.deformation.output_range_min) / 2
                deformations = deformations / mlp_output_scale * self._rays.norm(dim=-1, keepdim=True)
                
        return deformations
    
    @property
    def verts(self):
        return self._verts + self.verts_deformations.view(self.n_pm, self.pm_h, self.pm_w, 3)
    
    @property
    def deformed_depths(self):
        all_depths = []
        for i_depth in range(self.n_pm):
            _depth_i = self.cameras.p3d_cameras[i_depth].get_world_to_view_transform().transform_points(
                self._deformed_verts[i_depth].reshape(-1, 3)
            )[..., 2].reshape(self.pm_h, self.pm_w)
            all_depths.append(_depth_i[None])
        return torch.cat(all_depths, dim=0)
    
    @property
    def confidence(self):
        return 1. + torch.exp(self._confidence)  # (n_depth, h, w)
    
    def forward(self):
        pass
    
    def loss(self, reference_depths, pred_depths, masks=None):
        if self.using_pts_as_reference:
            reference_pts_deformed_depths = []
            for i_depth in range(self.n_pm):
                reference_pts_deformed_depth_i, fov_mask = get_points_depth_in_depthmap(
                    pts=self.reference_pts[i_depth], 
                    depthmap=pred_depths[i_depth], 
                    p3d_camera=self.cameras.p3d_cameras[i_depth]
                )
                reference_pts_deformed_depths.append(reference_pts_deformed_depth_i)
            reference_pts_deformed_depths = torch.cat(reference_pts_deformed_depths, dim=0).flatten()
            diff = reference_pts_deformed_depths - reference_depths
        else:
            diff = pred_depths - reference_depths
        
        diff = diff.abs()  # TODO: Should we divide by reference_depths.median()?
        
        if self.use_learnable_confidence:
            confidence = self.confidence
            
            if self.using_pts_as_reference:
                reference_pts_confidence = []
                for i_depth in range(self.n_pm):
                    reference_pts_confidence_i, _ = get_points_depth_in_depthmap(
                        pts=self.reference_pts[i_depth], 
                        depthmap=confidence[i_depth], 
                        p3d_camera=self.cameras.p3d_cameras[i_depth]
                    )
                    reference_pts_confidence.append(reference_pts_confidence_i)
                confidence = torch.cat(reference_pts_confidence, dim=0).flatten()
            
            diff = confidence * diff - self.confidence_weighting * torch.log(confidence)
        
        if masks is not None:
            diff = masks * diff
        
        return diff.mean()
    
    @torch.no_grad()
    def reset_encodings(self):
        if self.use_learnable_depth_encoding:
            self.depth_encoding = DepthEncoding(
                num_charts=self.depth_encoding.num_charts,
                num_bins=self.depth_encoding_params.n_bins,
                encoding_dim=self.depth_encoding_params.encoding_dim,
                initialization_range=self.depth_encoding_params.initialization_range,
            ).to(self.depth_encoding.encodings.device)
        
        if self.use_multi_res_charts_encoding:
            self.charts_encoding = MultiResChartsEncoding(
                num_charts=self.charts_encoding.num_charts,
                height=self.pm_h,
                width=self.pm_w,
                resolutions=self.charts_encoding.resolutions,
                encoding_dim_per_res=self.charts_encoding_params.encoding_dim_per_res,
                initialization_range=self.charts_encoding_params.initialization_range,
            ).to(self.charts_encoding.charts_encoding[0].encodings.device)
        else:
            self.charts_encoding = ChartsEncoding(
                num_charts=self.charts_encoding.num_charts,
                encoding_h=self.chart_h,
                encoding_w=self.chart_w,
                encoding_dim=self.charts_encoding_params.encoding_dim,
                initialization_range=self.charts_encoding_params.initialization_range,
            ).to(self.charts_encoding.encodings.device)
        
    @torch.no_grad()
    def reset_mlp(self, std=None):
        if self.use_meta_mlp:
            self.deformation = DeformationMetaMultiMLP(
                n_heads=self.n_pm,
                n_layer=self.deformation.n_layer,
                layer_size=self.deformation.layer_size,
                input_dim=self.deformation.input_dim,
                cond_dim=self.deformation.cond_dim,
                output_dim=self.deformation.output_dim,
                additional_input_dim=self.deformation.additional_input_dim,
                data_input_range_min=self.deformation.data_input_range_min,
                data_input_range_max=self.deformation.data_input_range_max,
                mlp_input_range_min=self.deformation.mlp_input_range_min,
                mlp_input_range_max=self.deformation.mlp_input_range_max,
                output_range_min=self.deformation.output_range_min,
                output_range_max=self.deformation.output_range_max,
                non_linearity=self.deformation.non_linearity,
                final_non_linearity=self.deformation.final_non_linearity,
                positional_encoding=self.deformation._positional_encoding,
            ).to(self.device)
            initialize_meta_multi_mlp_weights(self.deformation, std=std)
        elif self.use_lora_mlp:
            self.deformation = DeformationLoRAMultiMLP(
                n_heads=self.n_pm,
                n_layer=self.deformation.n_layer,
                layer_size=self.deformation.layer_size,
                input_dim=self.deformation.input_dim,
                output_dim=self.deformation.output_dim,
                cond_dim=self.deformation.cond_dim,
                lora_rank=self.deformation.lora_rank,
                n_lora_layers=self.deformation.n_lora_layers,
                additional_input_dim=self.deformation.additional_input_dim,
                data_input_range_min=self.deformation.data_input_range_min,
                data_input_range_max=self.deformation.data_input_range_max,
                mlp_input_range_min=self.deformation.mlp_input_range_min,
                mlp_input_range_max=self.deformation.mlp_input_range_max,
                output_range_min=self.deformation.output_range_min,
                output_range_max=self.deformation.output_range_max,
                non_linearity=self.deformation.non_linearity,
                final_non_linearity=self.deformation.final_non_linearity,
                positional_encoding=self.deformation._positional_encoding,
            ).to(self.device)
            initialize_lora_multi_mlp_weights(self.deformation, std=std)
        else:
            self.deformation = DeformationMultiMLP(
                n_heads=self.n_pm,
                n_layer=self.deformation.n_layer,
                layer_size=self.deformation.layer_size,
                input_dim=self.deformation.input_dim,
                output_dim=self.deformation.output_dim,
                additional_input_dim=self.deformation.additional_input_dim,
                data_input_range_min=self.deformation.data_input_range_min,
                data_input_range_max=self.deformation.data_input_range_max,
                mlp_input_range_min=self.deformation.mlp_input_range_min,
                mlp_input_range_max=self.deformation.mlp_input_range_max,
                output_range_min=self.deformation.output_range_min,
                output_range_max=self.deformation.output_range_max,
                non_linearity=self.deformation.non_linearity,
                final_non_linearity=self.deformation.final_non_linearity,
                positional_encoding=self.deformation._positional_encoding,
            ).to(self.device)
            initialize_multi_mlp_weights(self.deformation, std=std)
    
    def prepare_for_optimization(
        self,
        encodings_lr:float=1e-2,
        mlp_lr:float=1e-3,
        lr_update_iters=[300],
        lr_update_factor:float=0.1,
        confidence_lr:float=1e-3,
        verbose:bool=True,
        extrinsics_lr:float=1e-4,
        intrinsics_lr:float=1e-4,
    ):
        # Reset encodings and MLP
        self.reset_encodings()
        self.reset_mlp()
        # self.using_pts_as_reference = None
        
        # Initialize optimizer
        if self.use_multi_res_charts_encoding:
            l = []
            names = []
            for name, param in self.charts_encoding.named_parameters():
                full_name = f"charts_encoding.{name}"
                l.append({'params': [param], 'lr':encodings_lr, "name": full_name})
                names.append(full_name)
        else:
            l = [{'params': [self.charts_encoding.encodings], 'lr':encodings_lr, "name": "charts_encoding.encodings"},]
            names = ["charts_encoding.encodings"]
        
        if self.use_learnable_depth_encoding:
            l = l + [{'params': [self.depth_encoding.encodings], 'lr':encodings_lr, "name": "depth_encoding.encodings"}]
            names = names + ["depth_encoding.encodings"]
       
        if self.use_learnable_confidence:
            l = l + [{'params': [self._confidence], 'lr':confidence_lr, "name": "_confidence"}]
            names = names + ["_confidence"]
            
        if self.learnable_extrinsics:
            l = l + [{'params': [self.cameras.delta_quaternions], 'lr':extrinsics_lr, "name": "cameras.delta_quaternions"},
                      {'params': [self.cameras.delta_translations], 'lr':extrinsics_lr, "name": "cameras.delta_translations"}]
            names = names + ["cameras.delta_quaternions", "cameras.delta_translations"]
        
        if self.learnable_intrinsics:
            l = l + [{'params': [self.cameras.delta_focal_x], 'lr':intrinsics_lr, "name": "cameras.delta_focal_x"},
                      {'params': [self.cameras.delta_focal_y], 'lr':intrinsics_lr, "name": "cameras.delta_focal_y"},
                      {'params': [self.cameras.delta_prcppoint], 'lr':intrinsics_lr, "name": "cameras.delta_prcppoint"}]
            names = names + ["cameras.delta_focal_x", "cameras.delta_focal_y", "cameras.delta_prcppoint"]
        
        for name, param in self.named_parameters():
            if name in names:
                continue
            l.append({'params': [param], 'lr':mlp_lr, "name": name})
            names.append(name)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if verbose:
            print("=== Optimizable parameters ===")
            for param_group in self.optimizer.param_groups:
                print(param_group['name'], param_group['lr'])
                print("   > Shape:", param_group['params'][0].shape)
        
        self.lr_update_iters = lr_update_iters
        self.lr_update_factor = lr_update_factor
        
    def optimize(
        self,
        reference_data:torch.Tensor,
        masks:torch.Tensor=None,
        gradient_masks:torch.Tensor=None,
        n_iterations=600,
        use_gradient_loss=True,
        use_hessian_loss=False,
        use_normal_loss=True,
        use_curvature_loss=False,
        use_matching_loss=False,
        use_reprojection_loss=False,
        matching_thr=None,
        use_confidence_in_matching_loss=False,
        matching_update_iters=None,
        gradient_loss_weight=10.,
        hessian_loss_weight=100.,
        normal_loss_weight=2.,
        curvature_loss_weight=1.,
        matching_loss_weight=1.,
        reprojection_loss_weight=2.,
        encodings_lr:float=1e-2,
        mlp_lr:float=1e-3,
        confidence_lr:float=1e-3,
        lr_update_iters=[300],
        lr_update_factor:float=0.1,
        verbose:bool=True,
        match_to_img:torch.Tensor=None,
        match_to_pix:torch.Tensor=None,
        reprojection_loss_power:float=0.5,
        regularize_chart_encodings_norms:bool=False,
        chart_encodings_norm_loss_weight:float=0.5,
        use_total_variation_on_depth_encodings:bool=False,
        total_variation_on_depth_encodings_weight:float=1.0,
    ):
        """_summary_

        Args:
            reference_data (torch.Tensor): Should have shape (n_charts, h, w), 
                or can be a list or tensor containing n_charts tensors of shape (n_points_in_chart, 3).
            n_iterations (int, optional): _description_. Defaults to 600.
            use_gradient_loss (bool, optional): _description_. Defaults to True.
            use_hessian_loss (bool, optional): _description_. Defaults to False.
            gradient_loss_weight (_type_, optional): _description_. Defaults to 10..
            hessian_loss_weight (_type_, optional): _description_. Defaults to 100..
            encodings_lr (float, optional): _description_. Defaults to 1e-2.
            mlp_lr (float, optional): _description_. Defaults to 1e-3.
            lr_update_iters (list, optional): _description_. Defaults to [300].
            lr_update_factor (float, optional): _description_. Defaults to 0.1.
            verbose (bool, optional): _description_. Defaults to True.
        """
        train_losses = []
        if verbose:
            print(f"Starting optimization...")
            
        # Preprocessing of reference depths
        if reference_data[0].shape[-1] == 3:
            print("Using a list of 3D points as reference for fitting depth maps.")
            self.using_pts_as_reference = True
            try:
                if isinstance(reference_data, list):
                    reference_pts = torch.stack(reference_data, dim=0)
                reference_depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(reference_pts)[..., 2].flatten()
                print(f"\nReference points has shape {reference_pts.shape[0]}.")
                print(f"Reference points depths has shape {reference_depths.shape[0]}.")
                consistent_n_points = True
            except:
                print("Converting list of 3D points to tensor failed. The number of points in each chart is not consistent.")
                print("Each chart will be processed separately.")
                print("[TODO] Implement a simple padding mechanism to handle this case.")
                reference_pts = reference_data
                reference_depths = []
                for i_depth in range(len(reference_data)):
                    reference_pts_i = reference_pts[i_depth]
                    reference_pts_depth_i = self.cameras.p3d_cameras[i_depth].get_world_to_view_transform().transform_points(reference_pts_i)[..., 2]
                    reference_depths.append(reference_pts_depth_i)
                    print(f"\nDepth {i_depth} has {reference_pts_i.shape[0]} reference points.")
                reference_depths = torch.cat(reference_depths, dim=0).flatten()
                print(f"Reference points depths has shape {reference_depths.shape}.")
                consistent_n_points = False
            self.reference_pts = reference_pts
        else:
            print("Using depth maps as reference for fitting depth maps.")
            self.using_pts_as_reference = False
            reference_depths = reference_data
            if masks is not None:
                print("Using masks for optimization.")
                assert masks.shape == reference_depths.shape
            if gradient_masks is not None:
                print("Using gradient masks for optimization.")
                assert gradient_masks.shape == self._depths[..., :-1, :-1].shape
                
        # Prepare matcher if needed
        if use_matching_loss:
            if self.using_pts_as_reference:
                raise NotImplementedError("Matching loss is not implemented yet for point clouds.")
            matcher = Matcher3D(cameras=self.cameras, reference_depths=reference_depths)
            if matching_thr is None:
                matching_thr = self.cameras.get_spatial_extent() / 20.
            matcher.match(matching_thr)
            
        # Prepare for optimization
        self.prepare_for_optimization(
            encodings_lr=encodings_lr,
            mlp_lr=mlp_lr,
            confidence_lr=confidence_lr,
            lr_update_iters=lr_update_iters,
            lr_update_factor=lr_update_factor,
            verbose=verbose,
        )
        
        if use_normal_loss or use_curvature_loss:
            _normals = depth2normal_parallel(self._depths, self.cameras)
            self.initial_normals = _normals
        else:
            self.initial_normals = None
        if use_curvature_loss:
            _curvatures = normal2curv_parallel(_normals, mask=torch.ones_like(_normals, dtype=torch.bool))
            self.initial_curvatures = _curvatures
        else:
            self.initial_curvatures = None

        # Optimization loop
        progress_bar = tqdm(range(n_iterations), desc="Aligning charts")
        for i_iter in range(n_iterations):
            if i_iter in lr_update_iters:
                if verbose:
                    print("\n[INFO] Updating learning rates...")

                # Print previous learning rates
                if verbose:
                    print("   > Previous learning rates:")
                    for param_group in self.optimizer.param_groups:
                        print(f"      > {param_group['name']}: {param_group['lr']}")
                
                # Update learning rates
                for param_group in self.optimizer.param_groups:
                    if (param_group['name'].endswith('encodings')
                        or param_group['name'].startswith('deformation')
                    ):
                        param_group['lr'] = param_group['lr'] * lr_update_factor

                # Print updated learning rates
                if verbose:
                    print("   > Updated learning rates:")
                    for param_group in self.optimizer.param_groups:
                        print(f"      > {param_group['name']}: {param_group['lr']}")
            
            # Compute deformed depth
            _deformed_verts = self.verts
            _deformed_depths = self.cameras.p3d_cameras.get_world_to_view_transform().transform_points(
                _deformed_verts.reshape(self.n_pm, -1, 3)
            )[..., 2].reshape(self.n_pm, self.pm_h, self.pm_w)
            
            # Compute loss
            loss = self.loss(reference_depths=reference_depths, pred_depths=_deformed_depths, masks=masks)
            _loss = loss.detach().item()
            
            if use_gradient_loss:
                if gradient_masks is not None:
                    grad_loss = gradient_loss_weight * (gradient_masks * (img_grad(_deformed_depths) - img_grad(self._depths))).abs().mean()
                else:
                    grad_loss = gradient_loss_weight * (img_grad(_deformed_depths) - img_grad(self._depths)).abs().mean()
                loss += grad_loss
                grad_loss = grad_loss.detach().item()
            
            if use_hessian_loss:
                if gradient_masks is not None:                
                    hess_loss = hessian_loss_weight * (gradient_masks * (img_hessian(_deformed_depths) - img_hessian(self._depths))).abs().mean()
                else:
                    hess_loss = hessian_loss_weight * (img_hessian(_deformed_depths) - img_hessian(self._depths)).abs().mean()
                loss += hess_loss
                hess_loss = hess_loss.detach().item()
                
            if use_normal_loss:
                _deformed_normals = depth2normal_parallel(_deformed_depths, self.cameras)
                normal_loss = normal_loss_weight * (1. - torch.sum(_normals * _deformed_normals, dim=-1)).mean()
                loss += normal_loss
                normal_loss = normal_loss.detach().item()
                
            if use_curvature_loss:
                if not use_normal_loss:
                    _deformed_normals = depth2normal_parallel(_deformed_depths, self.cameras)
                _deformed_curvatures = normal2curv_parallel(_deformed_normals, mask=torch.ones_like(_deformed_normals, dtype=torch.bool))
                curv_loss = curvature_loss_weight * (_curvatures - _deformed_curvatures).abs().mean()
                loss += curv_loss
                curv_loss = curv_loss.detach().item()
            
            if use_matching_loss:
                reprojection_errors, fov_mask = matcher.compute_reprojection_errors(depths=_deformed_depths)
                reprojection_errors = reprojection_errors * fov_mask * matcher.reference_matches  # (n_charts, n_charts, h, w)
                if use_confidence_in_matching_loss:
                    reprojection_errors = reprojection_errors * self.confidence.detach()[None]  # (n_charts, n_charts, h, w)
                matching_loss = matching_loss_weight * reprojection_errors.mean()
                loss += matching_loss
                matching_loss = matching_loss.detach().item()
                
            if use_reprojection_loss and match_to_img is not None and match_to_pix is not None:
                minimal_projections_diffs = get_minimal_projections_diffs(
                    points3d=_deformed_verts,
                    cameras=self.cameras,
                    match_to_img=match_to_img,
                    match_to_pix=match_to_pix,
                    loss_power=reprojection_loss_power,
                )
                reprojection_loss = reprojection_loss_weight * minimal_projections_diffs.mean()
                loss += reprojection_loss
                reprojection_loss = reprojection_loss.detach().item()
                
            if regularize_chart_encodings_norms:
                chart_encodings_norm_loss = self.charts_encoding(self._pts_uv).norm(dim=-1).mean()
                loss += chart_encodings_norm_loss_weight * chart_encodings_norm_loss
                chart_encodings_norm_loss = chart_encodings_norm_loss.detach().item()
                
            if use_total_variation_on_depth_encodings:
                depth_encodings_tv_loss = (self.depth_encoding.encodings[..., 1:] - self.depth_encoding.encodings[..., :-1]).abs().mean()
                loss += total_variation_on_depth_encodings_weight * depth_encodings_tv_loss
                depth_encodings_tv_loss = depth_encodings_tv_loss.detach().item()
                
            # Update parameters
            loss.backward()
            
            # Optimization step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            
            # Update matchings if needed
            if use_matching_loss and (matching_update_iters is not None) and (i_iter in matching_update_iters):
                print("\n[INFO] Updating matchings.")
                matcher.update_references(reference_depths=_deformed_depths.detach())
                matcher.match(matching_thr)
            
            with torch.no_grad():
                iter_interval = n_iterations // 50
                if i_iter % iter_interval == 0:
                    train_losses.append(loss.detach().item())
                    if True:
                        loss_dict = {"loss": f"{loss.item():.5f}", "depth_loss": f"{_loss:.5f}"}
                        if use_gradient_loss:
                            loss_dict["grad"] = f"{grad_loss:.5f}"
                        if use_hessian_loss:
                            loss_dict["hess"] = f"{hess_loss:.5f}"
                        if use_normal_loss:
                            loss_dict["normal"] = f"{normal_loss:.5f}"
                        if use_curvature_loss:
                            loss_dict["curv"] = f"{curv_loss:.5f}"
                        if use_matching_loss:
                            loss_dict["matching"] = f"{matching_loss:.5f}"
                        if use_reprojection_loss:
                            loss_dict["reproj"] = f"{reprojection_loss:.5f}"
                        if regularize_chart_encodings_norms:
                            loss_dict["ce_norm"] = f"{chart_encodings_norm_loss:.5f}"
                        if use_total_variation_on_depth_encodings:
                            loss_dict["de_tv"] = f"{depth_encodings_tv_loss:.5f}"
                        progress_bar.update(iter_interval)
                        progress_bar.set_postfix(loss_dict)
                    else:
                        if verbose:
                            print(f"Iteration {i_iter}: Loss = {loss.item()}")
                            print(f"   > Depth loss = {_loss}")
                            if use_gradient_loss:
                                print(f"   > Gradient Loss = {grad_loss}")
                            if use_hessian_loss:
                                print(f"   > Hessian Loss = {hess_loss}")
                            if use_normal_loss:
                                print(f"   > Normal Loss = {normal_loss}")
                            if use_curvature_loss:
                                print(f"   > Curvature Loss = {curv_loss}")
                            if use_matching_loss:
                                print(f"   > Matching Loss = {matching_loss}")
                            if use_reprojection_loss:
                                print(f"   > Reprojection Loss = {reprojection_loss}")
                            if regularize_chart_encodings_norms:
                                print(f"   > Chart encodings norms loss = {chart_encodings_norm_loss}")
                            if use_total_variation_on_depth_encodings:
                                print(f"   > Total variation on depth encodings loss = {depth_encodings_tv_loss}")
                            for name, param in self.named_parameters():
                                if param.requires_grad:
                                    print(f"      > {name}")
                                    print(f"         > Max: {param.max().item()}   Min: {param.min().item()}   Mean: {param.mean().item()}   Std: {param.std().item()}")
        with torch.no_grad():
            self._deformed_verts[...] = self.verts.clone()
        progress_bar.close()
        if verbose:
            print("Optimization done.")
        self.train_losses = train_losses
    
    @torch.no_grad()
    def save_point_cloud(self, path, depth_idx=None, verbose=True):
        if depth_idx is None:
            xyz = self._deformed_verts.view(-1, 3)
        else:
            xyz = self._deformed_verts[depth_idx].view(-1, 3)
            
        pc = o3d.geometry.PointCloud()
        if verbose:
            print("Saving pointcloud with ", len(xyz), " points.")
        pc.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
        o3d.io.write_point_cloud(path, pc)

        if verbose:
            print(f"Saved at path: {path}")
        
        return xyz
    
    @torch.no_grad()
    def save_charts_data(
        self, 
        path:str,
        images:torch.Tensor, 
        conf_th:float=0.,
        ratio_th:float=5.,
        scale_factor:float=1.,
    ):
        # Output confidence
        output_confs = self.confidence
        
        # Output points
        output_verts = self._deformed_verts.clone()
        if images is None:
            images = torch.ones_like(output_verts)
        
        pm_h, pm_w = output_verts.shape[1:3]
        full_mesh = get_manifold_meshes_from_pointmaps(
            points3d=output_verts,
            imgs=images, 
            masks=output_confs > conf_th,  
            return_single_mesh_object=True
        )

        faces_verts = full_mesh.verts_packed()[full_mesh.faces_packed()]  # (n_faces, 3, 3)
        sides = (
            torch.roll(faces_verts, 1, dims=1)  # C, A, B
            - faces_verts  # A, B, C
        )  # (n_faces, 3, 3)  ;  AC, BA, CB
        normalized_sides = torch.nn.functional.normalize(sides, dim=-1)  # (n_faces, 3, 3)  ;  AC/||AC||, BA/||BA||, CB/||CB||
        alts = (
            sides  # AC
            - (sides * torch.roll(normalized_sides, -1, dims=1)).sum(dim=-1, keepdim=True) * normalized_sides # - (AC . BA) BA / ||BA||^2
        )  # (n_faces, 3, 3)
        alt_lengths = alts.norm(dim=-1)
        alt_ratios = alt_lengths.max(dim=1).values / alt_lengths.min(dim=1).values
        faces_mask = alt_ratios < ratio_th
        full_mesh = remove_faces_from_single_mesh(full_mesh, faces_to_keep_mask=faces_mask)
        output_pts = full_mesh.verts_packed()
        output_col = full_mesh.textures.verts_features_packed()
        
        # Output depths
        output_depths = torch.cat([
            self.cameras.p3d_cameras[i_chart].get_world_to_view_transform().transform_points(
            output_verts[i_chart].reshape(-1, 3)
        )[..., 2].reshape(1, pm_h, pm_w) for i_chart in range(len(self.cameras))
        ], dim=0)

        # Prior depths
        if self._depths is not None:
            depths = torch.cat([
                self._depths[i_chart].reshape(1, pm_h, pm_w) for i_chart in range(len(self.cameras))
            ], dim=0)
        else:
            depths = None

        # Prior normals
        if self.initial_normals is not None:
            normals = torch.cat([
                self.initial_normals[i_chart].reshape(1, pm_h, pm_w) for i_chart in range(len(self.cameras))
            ], dim=0)
        else:
            normals = None

        # Prior curvatures
        if self.initial_curvatures is not None:
            curvatures = torch.cat([
                self.initial_curvatures[i_chart].reshape(1, pm_h, pm_w) for i_chart in range(len(self.cameras))
            ], dim=0)
        else:
            curvatures = None
            
        # Extrinsics
        additional_kwargs = {}
        
        if self.learnable_extrinsics:
            additional_kwargs["quaternions"] = self.cameras.delta_quaternions
            additional_kwargs["translations"] = self.cameras.delta_translations
            
        # Intrinsics
        if self.learnable_intrinsics:
            additional_kwargs["focal_x"] = self.cameras.delta_focal_x
            additional_kwargs["focal_y"] = self.cameras.delta_focal_y
            additional_kwargs["prcppoint"] = self.cameras.delta_prcppoint

        if path is not None:
            np.savez(
                path, 
                pts=(output_pts * scale_factor).cpu().numpy(), 
                cols=output_col.cpu().numpy(), 
                confs=output_confs.cpu().numpy(), 
                depths=(output_depths).cpu().numpy(), 
                normals=normals.cpu().numpy(), 
                curvatures=curvatures.cpu().numpy(),
                **additional_kwargs
            )
