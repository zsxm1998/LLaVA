import torch
import torch.nn as nn
from contextlib import nullcontext

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, load_parameter_first=True):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.tune_vision_tower = getattr(args, 'tune_vision_tower', False)
        if isinstance(self.select_layer, str):
            self.select_layer = [int(part) for part in self.select_layer.split(',')]
            if len(self.select_layer) == 1:
                self.select_layer = self.select_layer[0]
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model(load_parameter_first)
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, load_parameter=True):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if load_parameter:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        else:
            self.vision_tower = CLIPVisionModel(CLIPVisionConfig.from_pretrained(self.vision_tower_name))
        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if isinstance(self.select_layer, list):
            feature_list = []
            for i in self.select_layer:
                image_features = image_forward_outs.hidden_states[i]
                if self.select_feature == 'patch':
                    image_features = image_features[:, 1:]
                elif self.select_feature == 'cls_patch':
                    image_features = image_features
                else:
                    raise ValueError(f'Unexpected select feature: {self.select_feature}')
                feature_list.append(image_features)
            return torch.stack(feature_list, dim=0) #[N, B, L, D]
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
            elif self.select_feature == 'cls_patch':
                image_features = image_features
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            return image_features #[B, L, D]

    def forward(self, images):
        with nullcontext() if self.tune_vision_tower else torch.no_grad():
            if type(images) is list:
                image_features = []
                for image in images:
                    image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

            return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        if self.config.mm_projector_type.lower().startswith('qformer'):
            return int(self.config.mm_projector_type.split('_')[1])
        return (self.config.image_size // self.config.patch_size) ** 2
