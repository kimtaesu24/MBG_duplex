#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import numpy as np
import torch.nn as nn
from pytorch3d.io import load_obj
from pytorch3d.structures import (
    Meshes, Pointclouds
)
from pytorch3d.renderer import (
    PerspectiveCameras, BlendParams,
    look_at_view_transform, RasterizationSettings,
    FoVPerspectiveCameras, PointsRasterizationSettings,
    PointsRenderer, PointsRasterizer, AlphaCompositor,
    PointLights, AmbientLights, TexturesVertex, TexturesUV,
    HardGouraudShader, HardPhongShader, SoftPhongShader, SoftSilhouetteShader,
    MeshRasterizer, MeshRenderer, Materials
)


class RenderMesh(nn.Module):
    def __init__(self, image_size, obj_filename=None, faces=None, scale=1.0):
        super(RenderMesh, self).__init__()
        self.ori_size = image_size
        image_size = int(image_size)
        self.scale = scale
        self.image_size = image_size
        if obj_filename is not None:
            verts, faces, aux = load_obj(obj_filename, load_textures=False)
            self.faces = faces.verts_idx
        elif faces is not None:
            import numpy as np
            if isinstance(faces, torch.Tensor):
                self.faces = faces
            else:
                self.faces = torch.tensor(faces.astype(np.int32))
        else:
            raise NotImplementedError('Must have faces.')
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)

    def _build_cameras(self, transform_matrix, focal_length, device):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=device).float(), 'focal_length': focal_length,
            'image_size': screen_size, 'device': device,
        }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    @torch.no_grad()
    def forward(self, vertices, cameras=None, transform_matrix=None, focal_length=None, colors=None):
        if cameras is None and transform_matrix is not None:
            cameras = self._build_cameras(transform_matrix, focal_length, device=vertices.device)
        if cameras is None and transform_matrix is None:
            transform_matrix = torch.tensor(
                [[[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 2*self.scale]]],
                dtype=torch.float32, device=vertices.device
            )
            cameras = self._build_cameras(transform_matrix, 12.0, device=vertices.device)
        faces = self.faces[None].repeat(vertices.shape[0], 1, 1)
        # Initialize each vertex to be white in color.
        if colors is None:
            verts_rgb = torch.ones_like(vertices) * torch.tensor([142, 179, 247])[None, None].type_as(vertices) / 255.0  # (1, V, 3)
        else:
            verts_rgb = (colors.type_as(vertices)[None] / 255.0).clamp(0.0, 1.0).expand(vertices.shape[0], -1, -1)
        textures = TexturesVertex(verts_features=verts_rgb.to(vertices.device))
        mesh = Meshes(
            verts=vertices.to(vertices.device), faces=faces.to(vertices.device), textures=textures
        )
        lights = PointLights(location=[[0.0, 1.0, 3.0]], device=vertices.device)
        materials = Materials(
            device=vertices.device, specular_color=[[0.6, 0.6, 0.6]], shininess=10.0
        )
        blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=HardPhongShader(cameras=cameras, materials=materials, lights=lights, blend_params=blend_params, device=vertices.device)
        )
        render_results = renderer(mesh).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        images = torch.nn.functional.interpolate(images, (self.ori_size, self.ori_size), mode='area')
        return images*255, alpha_images
