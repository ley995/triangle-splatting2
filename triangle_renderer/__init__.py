#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import math
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from scene.triangle_model import TriangleModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
import torch.nn.functional as F

def render(viewpoint_camera, pc : TriangleModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    
    triangles_indices = pc.get_triangle_indices  # the idx of the 3 vertices of each triangl
    vertices = pc.get_vertices # contains all the vertices of the triangles
    vertex_weights = pc.get_vertex_weight  # contains the weights of the vertices for each vertex in the triangles
    scaling = torch.zeros_like(triangles_indices[:, 0], dtype=pc.get_triangles_points.dtype, requires_grad=True, device="cuda").detach()



    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H_init = int(viewpoint_camera.image_height)
    W_init = int(viewpoint_camera.image_width)

    upsample = pc.scaling

    H = upsample * H_init
    W = upsample * W_init

    raster_settings = TriangleRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    sigma = pc.get_sigma

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    else:
        colors_precomp = override_color

    # Rasterize visible triangles to image, obtain their radii (on screen). 
    rendered_image, radii, scaling, allmap, max_blending  = rasterizer(
        vertices=vertices,
        triangles_indices=triangles_indices,
        vertex_weights=vertex_weights.squeeze(),
        sigma=sigma,
        shs = shs,
        colors_precomp = colors_precomp,
        scaling = scaling
       )
       
    img_hr = rendered_image.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    rendered_image = img_ds_area.squeeze(0)
    

    rets =  {"render": rendered_image,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            "scaling": scaling,
            "max_blending": max_blending,
            }

    # additional regularizations
    render_alpha = allmap[1:2]
    img_hr = render_alpha.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    render_alpha = img_ds_area.squeeze(0)

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    img_hr = render_normal.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    render_normal = img_ds_area.squeeze(0)
    #render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    img_hr = render_depth_median.unsqueeze(0)  # -> [1, 3, H, W]
    img_ds_area = F.interpolate(img_hr, size=(H_init, W_init), mode="area")  # [1, 3, H0, W0]
    render_depth_median = img_ds_area.squeeze(0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    rendered_entropy = allmap[0:1]
    
    
    # gets the id per pixel of the triangle influencing it
    render_id = allmap[6:7]
    
    surf_depth = render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_ids': render_id,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            "entropy": rendered_entropy,
            "depth_full": allmap[5:6]
    })

    return rets





 