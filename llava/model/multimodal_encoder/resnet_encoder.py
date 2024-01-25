import torch
import torch.nn as nn

from transformers import ResNetModel, CLIPImageProcessor, ResNetConfig


class ResNetVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ResNetConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = ResNetModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch':
            image_features = image_forward_outs.hidden_states + (image_forward_outs.pooler_output,)
            return image_features[self.select_layer]
        elif self.select_feature == 'pool':
            return image_forward_outs.pooler_output
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}, expecting "patch" or "pool"')

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, return_dict=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype).flatten(2).transpose(1,2)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, return_dict=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype).flatten(2).transpose(1,2)

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
        hs_list = [self.config.embedding_size] + self.config.hidden_sizes + self.config.hidden_sizes[-1:]
        return hs_list[self.select_layer]

    @property
    def num_patches(self):
        if self.select_feature == 'pool' or self.select_layer in [-1, 5]:
            return 1
        down_list = torch.tensor([4, 2 if self.config.downsample_in_first_stage else 1, 2, 2, 2, 1], dtype=torch.long).cumprod(0).tolist()
        down_ratio = down_list[self.select_layer]
        image_edge = (self.image_processor.crop_size+down_ratio-1) // down_ratio
        return image_edge ** 2
