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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
import math
from pytorch3d.ops import knn_points
import triangulation
#import igl



class TriangleModel:

    def setup_functions(self):
        """self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid"""

        self.eps = 1e-6
        self.opacity_floor = 0.0
        self.opacity_activation = lambda x: self.opacity_floor + (1.0 - self.opacity_floor) * torch.sigmoid(x)
        # Matching inverse for any y in [m, 1): logit( (y - m)/(1 - m) )
        self.inverse_opacity_activation = lambda y: inverse_sigmoid(
            ((y.clamp(self.opacity_floor + self.eps, 1.0 - self.eps) - self.opacity_floor) /
            (1.0 - self.opacity_floor + self.eps))
        )

        self.exponential_activation = lambda x:math.exp(x)
        self.inverse_exponential_activation = lambda y: math.log(y)

    def __init__(self, sh_degree : int):

        self._triangles = torch.empty(0) # can be deleted eventually

        self.size_probs_zero = 0.0
        self.size_probs_zero_image_space = 0.0
        self.vertices = torch.empty(0)
        self._triangle_indices = torch.empty(0)
        self.vertex_weight = torch.empty(0)

        self._sigma = 0
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.optimizer = None
        self.image_size = 0
        self.importance_score = 0
        self.add_percentage = 1.0

        self.scaling = 1

        self.setup_functions()

    def save_parameters(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self.vertices
        point_cloud_state_dict["_triangle_indices"] = self._triangle_indices
        point_cloud_state_dict["vertex_weight"] = self.vertex_weight
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
        point_cloud_state_dict["features_dc"] = self._features_dc
        point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["importance_score"] = self.importance_score
        point_cloud_state_dict["image_size"] = self.image_size

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))

   

    def load_parameters(self, path, device="cuda", segment=False, ratio_threshold = 0.25):
        # 1. Load the dict you saved
        state = torch.load(os.path.join(path, "point_cloud_state_dict.pt"), map_location=device)

        # 2. Restore everything you put in there (one line each)
        self.vertices            = state["triangles_points"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._triangle_indices   = state["_triangle_indices"].to(device).to(torch.int32)
        self.vertex_weight       = state["vertex_weight"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma              = state["sigma"]
        self.active_sh_degree    = state["active_sh_degree"]
        self._features_dc        = state["features_dc"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest      = state["features_rest"].to(device).to(torch.float32).detach().clone().requires_grad_(True)
        self.importance_score = state["importance_score"].to(device).to(torch.float32).detach().clone().requires_grad_(True)


        # For object extraction
        if segment:
            base = os.path.dirname(os.path.dirname(path))
            triangle_hits = torch.load(os.path.join(base, 'segmentation/triangle_hits_mask.pt'))
            triangle_hits_total = torch.load(os.path.join(base, 'segmentation/triangle_hits_total.pt'))

            min_hits = 1  

            # Handle division by zero - triangles with no renders get ratio 0
            triangle_ratio = torch.zeros_like(triangle_hits, dtype=torch.float32)
            valid_mask = triangle_hits_total > 0
            triangle_ratio[valid_mask] = triangle_hits[valid_mask].float() / triangle_hits_total[valid_mask].float()

            # Create the keep mask: triangles must meet both ratio and minimum hits criteria
            keep_mask = (triangle_ratio >= ratio_threshold) & (triangle_hits >= min_hits)
            #keep_mask = ~keep_mask

            with torch.no_grad():
                self._triangle_indices = self._triangle_indices[keep_mask]

        ################################################################


        self.opacity_floor = 0.9999

        # 3. (Re)compute any derived quantities

        self._triangle_indices = self._triangle_indices.to(torch.int32)


        param_groups = [
            {'params': [self._features_dc], 'lr': 0.0, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.0 / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': 0.0, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': 0.0, "name": "vertex_weight"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        opt_dict) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property
    def get_triangle_indices(self):
        return self._triangle_indices

    @property
    def get_triangles_points_flatten(self):
        return self._triangles.flatten(0)
  
    @property
    def get_triangles_points(self):
        return self._triangles

    @property
    def get_vertices(self):
        return self.vertices
    
    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_vertex_weight(self):
        return self.opacity_activation(self.vertex_weight)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def create_from_pcd(self, pcd : BasicPointCloud, opacity : float, set_sigma : float):

        # we remove all points that are too close to each other. Otherwise, this somehow gives an oom
        pcd_points = np.asarray(pcd.points)
        pcd_points = np.round(pcd_points, decimals=6)
        _, unique_indices = np.unique(pcd_points, axis=0, return_index=True)
        pcd_points = pcd_points[np.sort(unique_indices)]
        _points = torch.tensor(pcd_points).float().cuda()

        n = _points.shape[0]
        
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)[:n]).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        ################################
        # RUN DELAUNAY TRIANGULATION   #
        ################################
        dt = triangulation.Triangulation(_points)
        perm = dt.permutation().to(torch.long)
        _points = _points[perm]
        features = features[perm]
        
        tets = dt.tets()     
        tets = dt.tets().to(torch.int64)

        faces = torch.cat([
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ], dim=0)  # [4 * num_tets, 3]

        
        # Step 3: Sort to ignore winding order
        faces, _ = torch.sort(faces, dim=1)
        faces = torch.unique(faces, dim=0)

        # finally stash on your module
        self.vertices          = nn.Parameter(_points.requires_grad_(True))
        self._triangle_indices = faces.to(torch.int32)                         # [T,3]
        vert_weight = inverse_sigmoid(opacity * torch.ones((self.vertices.shape[0], 1), dtype=torch.float, device="cuda")) 
        self.vertex_weight = nn.Parameter(vert_weight.requires_grad_(True))

        # Sigma should be very low such that the triangles are solid. No need for soft triangles.
        self._sigma = self.inverse_exponential_activation(set_sigma)


        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        

    def training_setup(self, training_args, lr_mask, lr_features, weight_lr, lr_sigma, lr_triangles_init):

        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)




    def training_setup(self, training_args, lr_features, weight_lr, lr_triangles_init):

        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self.vertices], 'lr': lr_triangles_init, "name": "vertices"},
            {'params': [self.vertex_weight], 'lr': weight_lr, "name": "vertex_weight"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_init,
                                                        lr_final=lr_triangles_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def set_sigma(self, sigma):
        self._sigma = self.inverse_exponential_activation(sigma)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                    if iteration < 1000:
                        lr = 0
                    else:
                        lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    

    def densification_postfix(self, new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles):
        # Create dictionary of new tensors to append
        d = {
            "vertices": new_vertices,
            "vertex_weight": new_vertex_weight,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
        }
        
        # Append new tensors to optimizer
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # Update model parameters
        self.vertices = optimizable_tensors["vertices"]
        self.vertex_weight = optimizable_tensors["vertex_weight"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        
        # Update triangle indices
        self._triangle_indices = torch.cat([
            self._triangle_indices, 
            new_triangles
        ], dim=0)

        self.image_size = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((self._triangle_indices.shape[0]), dtype=torch.float, device="cuda")



    def _update_params_fast(self, selected_indices, iteration):
        selected_indices = torch.unique(selected_indices)
        selected_triangles_indices = self._triangle_indices[selected_indices]  # [S, 3]
        S = selected_triangles_indices.shape[0]
        
        edges = torch.cat([
            selected_triangles_indices[:, [0, 1]],
            selected_triangles_indices[:, [0, 2]],
            selected_triangles_indices[:, [1, 2]]
        ], dim=0) 
        edges_sorted, _ = torch.sort(edges, dim=1)
        
        unique_edges_tensor, unique_indices = torch.unique(
            edges_sorted, return_inverse=True, dim=0
        )  
        M = unique_edges_tensor.shape[0]
        
        v0 = self.vertices[unique_edges_tensor[:, 0]]
        v1 = self.vertices[unique_edges_tensor[:, 1]]
        new_vertices = (v0 + v1) / 2.0
        
        new_vertex_base = self.vertices.shape[0]
        
        unique_edges_cpu = unique_edges_tensor.cpu()
        edge_to_midpoint = {}
        for i in range(M):
            edge_tuple = (unique_edges_cpu[i, 0].item(), unique_edges_cpu[i, 1].item())
            edge_to_midpoint[edge_tuple] = new_vertex_base + i

        new_triangles_list = []
        selected_triangles_cpu = selected_triangles_indices.cpu()
        
        for i in range(S):
            tri = selected_triangles_cpu[i]
            a, b, c = tri[0].item(), tri[1].item(), tri[2].item()
            
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            
            m_ab = edge_to_midpoint[ab]
            m_ac = edge_to_midpoint[ac]
            m_bc = edge_to_midpoint[bc]

            new_triangles_list.append([a, m_ab, m_ac])
            new_triangles_list.append([b, m_ab, m_bc])
            new_triangles_list.append([c, m_ac, m_bc])
            new_triangles_list.append([m_ab, m_bc, m_ac])
        
        subdivided_triangles = torch.tensor(
            new_triangles_list, 
            dtype=torch.int32, 
            device=self._triangle_indices.device
        )

        u, v = unique_edges_tensor[:, 0], unique_edges_tensor[:, 1]
        new_features_dc = (self._features_dc[u] + self._features_dc[v]) / 2.0
        new_features_rest = (self._features_rest[u] + self._features_rest[v]) / 2.0
        
        opacity_u = self.opacity_activation(self.vertex_weight[u])
        opacity_v = self.opacity_activation(self.vertex_weight[v])
        avg_opacity = (opacity_u + opacity_v) / 2.0
        avg_opacity = torch.clamp(avg_opacity, self.opacity_floor + self.eps, 1 - self.eps)
        new_vertex_weight = self.inverse_opacity_activation(avg_opacity)

        new_triangles = subdivided_triangles
        
        return (
            new_vertices,
            new_vertex_weight,
            new_features_dc,
            new_features_rest,
            new_triangles
        )


    def _prune_vertex_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["vertices", "vertex_weight", "f_dc", "f_rest"]:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    # Prune optimizer state
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    
                    del self.optimizer.state[group['params'][0]]
                    # Update parameter
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group['params'][0]
        
        # Update model parameters
        for name, tensor in optimizable_tensors.items():
            if name == "vertices":
                self.vertices = tensor
            elif name == "vertex_weight":
                self.vertex_weight = tensor
            elif name == "f_dc":
                self._features_dc = tensor
            elif name == "f_rest":
                self._features_rest = tensor


    def _prune_vertices(self, vertex_mask: torch.Tensor):
        device = vertex_mask.device
        oldV = vertex_mask.numel()

        # Create mapping from old vertex IDs to new IDs (-1 for removed vertices)
        new_id = torch.full((oldV,), -1, dtype=torch.long, device=device)
        kept = torch.nonzero(vertex_mask, as_tuple=True)[0]
        new_id[kept] = torch.arange(kept.numel(), device=device, dtype=torch.long)

        # Remap triangle indices and drop triangles with removed vertices
        if self._triangle_indices.numel() > 0:
            remapped = new_id[self._triangle_indices.long()]
            valid_tris = (remapped >= 0).all(dim=1)
            remapped = remapped[valid_tris]
            self._triangle_indices = remapped.to(torch.int32).contiguous()

            if isinstance(self.image_size, torch.Tensor) and self.image_size.numel() > 0:
                self.image_size = self.image_size[valid_tris]
            if isinstance(self.importance_score, torch.Tensor) and self.importance_score.numel() > 0:
                self.importance_score = self.importance_score[valid_tris]

        # Prune vertex-related parameters using the initial mask
        self._prune_vertex_optimizer(vertex_mask)

        # After initial pruning, check for unreferenced vertices
        current_vertex_count = self.vertices.shape[0]
        if current_vertex_count > 0:
            # Identify vertices still referenced by triangles
            if self._triangle_indices.numel() > 0:
                referenced_vertices = torch.unique(self._triangle_indices)
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)
                mask_referenced[referenced_vertices] = True
            else:
                mask_referenced = torch.zeros(current_vertex_count, dtype=torch.bool, device=device)

            # Remove unreferenced vertices
            if not mask_referenced.all():
                # Prune vertex parameters
                self._prune_vertex_optimizer(mask_referenced)

                # Remap triangle indices if triangles exist
                if self._triangle_indices.numel() > 0:
                    new_id2 = torch.full((current_vertex_count,), -1, dtype=torch.long, device=device)
                    kept2 = torch.nonzero(mask_referenced, as_tuple=True)[0]
                    new_id2[kept2] = torch.arange(kept2.numel(), device=device, dtype=torch.long)
                    self._triangle_indices = new_id2[self._triangle_indices.long()].to(torch.int32).contiguous()





    def prune_triangles(self, mask):

        ##################################################################
        # REMOVE ALL TRIANLGES WITH MIN WEIGHT LESS THAN SOME THRESHOLD  #
        ##################################################################
        self._triangle_indices = self._triangle_indices[mask]

        #################################################
        # WE SET THE CLASS VARIABLES TO THEIR NEW VALUES#
        #################################################
        self._triangle_indices = self._triangle_indices.to(torch.int32)

        self.image_size = self.image_size[mask]
        self.importance_score = self.importance_score[mask]
        

    def _sample_alives(self, probs, num, alive_indices=None):
        torch.manual_seed(1)  # always same "random" indices
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=False)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        return sampled_idxs        

    def add_new_gs(self, iteration, cap_max, splitt_large_triangles, probs_opacity=True):

        current_num_points = self.vertices.shape[0]
        target_num = min(cap_max, int(self.add_percentage * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        # Find indexes based on proba
        triangle_transp = self.importance_score
        probs = triangle_transp.squeeze()

        areas = self.triangle_areas().squeeze()
        probs = torch.where(areas < self.size_probs_zero, torch.zeros_like(probs), probs)
        probs = torch.where(self.image_size < self.size_probs_zero_image_space, torch.zeros_like(probs), probs) # dont splitt if smaller than 10

        rand_idx = self._sample_alives(probs=probs, num=num_gs)

        # Split the largest triangles
        split_large = splitt_large_triangles
        k = min(split_large, areas.numel())  
        _, top_idx = torch.topk(areas, k, largest=True, sorted=False)

        # 3) combine and deduplicate
        add_idx = torch.unique(torch.cat([rand_idx, top_idx.to(rand_idx.device)]), sorted=False)

        (new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles) = self._update_params_fast(add_idx, iteration)

        self.densification_postfix(new_vertices, new_vertex_weight, new_features_dc, new_features_rest, new_triangles)

        mask = torch.ones(self._triangle_indices.shape[0], dtype=torch.bool)
        mask[add_idx] = False
        self.prune_triangles(mask)



    def update_min_weight(self, new_min_weight: float, preserve_outputs: bool = True):
        new_m = float(max(0.0, min(new_min_weight, 1.0 - 1e-4)))

        # 1) grab the current realized opacities y (under the old floor)
        with torch.no_grad():
            y = self.get_vertex_weight.detach()
            y = y.clamp(new_m + self.eps, 1.0 - self.eps)   # clamp to the *new* floor
        self.opacity_floor = new_m
        new_logits = self.inverse_opacity_activation(y)
        with torch.no_grad():
            self.vertex_weight.data.copy_(new_logits)


    def triangle_areas(self):
        tri = self.vertices[self._triangle_indices]                    # [T, 3, 3]
        AB  = tri[:, 1] - tri[:, 0]                                    # [T, 3]
        AC  = tri[:, 2] - tri[:, 0]                                    # [T, 3]
        cross_prod = torch.cross(AB, AC, dim=1)                        # [T, 3]
        areas = 0.5 * torch.linalg.norm(cross_prod, dim=1)             # [T]
        areas = torch.nan_to_num(areas, nan=0.0, posinf=0.0, neginf=0.0)
        return areas

