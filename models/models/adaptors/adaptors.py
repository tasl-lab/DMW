import os
from typing import Dict, List, Optional, Tuple, Union

from torch.distributions import Categorical

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
from user_profile_code.data_loader import load_all_user_data
from user_profile_code.profile_encoder import DeclaredProfileEncoder

from models.utils.custom_types import DrivingExample

def cross_track_error(points: Tensor, path: Tensor):
    """
    Computes the cross track error between a set of points and a path.

    Args:
        points: The set of points to compute the cross track error for with shape [b, n, 2].
        path: The path to compute the cross track error with with shape [b, m, 2]. The path
            can contain nan values which indicates that the path is not available for that position.

    Returns:
        The cross track error for each point in the set of points with shape [b, n].
    """

    points, path = points.float(), path.float()

    ind = torch.arange(path.size(0), device=path.device)[:, None]
    closest = torch.cdist(points, path).nan_to_num_(torch.inf).argmin(-1)
    pt0 = path[ind, (closest - 1).clamp_min(0)]
    pt1 = path[ind, closest]
    pt2 = path[ind, (closest + 1).clamp_max(path.size(1) - 1)]

    tangent = (pt2 - pt1).nan_to_num_(0.0) + (pt1 - pt0).nan_to_num_(0.0)
    normal = torch.stack((tangent[..., 1], -tangent[..., 0]), dim=-1)
    normal = normal / normal.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-2)

    return (points - pt1).mul(normal).sum(-1).abs()

class NormZeroOne(nn.Module):
    def __init__(self, min_max: Tuple[float, float]):
        super().__init__()
        self.register_buffer("min_max", torch.tensor(min_max, dtype=torch.float), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Normalise tensor to [0, 1] using values from min_max"""
        return (x - self.min_max[0]) / (self.min_max[1] - self.min_max[0])
    
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WaypointInputAdaptor(nn.Module):
    """
    Takes an input of shape [B, N, 2] and returns an output of shape [B, N, token_size]
    Args:
        token_size: feature dimension of output tensor.
        hidden_size: hidden dimension used in Linear layers under the hood.
        norm_layer: the `Module` to use to normalize the values of the input tensor.
    """
    
    def __init__(
        self, token_size: int = 258, hidden_size: int = 64, hidden_size2: int = 128, norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_layer = norm_layer

        self.mlp = nn.Sequential(nn.Linear(2, hidden_size), nn.ReLU(True), nn.Linear(hidden_size, hidden_size2), nn.ReLU(True), nn.Linear(hidden_size2, token_size))

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input with dims [B, N, 2]

        Returns:
            Output with dims [B, N, token_size]
        """
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.mlp(x)
        return x

class DreamingActionDiscreteHead(nn.Module):
    """
    Maps VLM features to a discrete joint action space.
    Returns a Categorical distribution over all (speed_scale, steer_delta) combinations.
    """
    def __init__(self, hidden_size: int, mlp_dim: int = 256, initial_temperature: float = 1.0, use_user_profile: bool = False):
        super().__init__()

        speed_list = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        steer_list = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

        # Learnable temperature (clamped to positive values)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(initial_temperature))
        )

        self.num_speed = len(speed_list)
        self.num_steer = len(steer_list)

        # register lookup tables as buffers (move with .to(device))
        speed = torch.tensor(speed_list, dtype=torch.float32)
        steer = torch.tensor(steer_list, dtype=torch.float32)
        grid_speed = speed.repeat_interleave(self.num_steer)
        grid_steer = steer.repeat(self.num_speed)
        actions_table = torch.stack([grid_speed, grid_steer], dim=-1)
        if not use_user_profile:
            zero_action = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
            actions_table = torch.cat([zero_action, actions_table], dim=0)

        self.register_buffer("actions_table", actions_table)
        num_actions = actions_table.size(0)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.SiLU(True),
            nn.Linear(mlp_dim, num_actions, bias=False)
        )

    def decode_indices(self, indices: Tensor) -> Tensor:
        """
        Maps discrete action indices to action values (speed_scale, steer_delta).
        Returns: [..., 2]
        """
        return self.actions_table[indices]

    def forward(self, features: Tensor, temperature: float = None) -> Dict[str, Union[Tensor, Categorical]]:
        """
        Args:
            features: [B, H] or [B, T, H] (sequence features are mean-pooled over dim=1)
        Returns:
            dict with keys 'dreaming_logits', 'dreaming_dist', 'actions_table'
        """
        x = features
        if x.dim() == 3:
            x = x.mean(dim=1, keepdim=True)

        logits = self.mlp(x)

        if temperature is None:
            temperature = torch.exp(self.log_temperature)
        
        dist = Categorical(logits=logits/temperature)

        return {
            "dreaming_logits": logits,
            "dreaming_dist": dist,
            "actions_table": self.actions_table
        }

class DrivingAdaptor(nn.Module):
    def __init__(self, 
                hidden_size: int, 
                mlp_dim=256, 
                predict_route_as_wps=False, 
                speed_wps_mode=False,
                rl_mode=False,
                use_user_profile=False,
            ):
        super().__init__()
        self.heads = {}
        self.order = []

        self.speed_wps_mode = speed_wps_mode
        self.predict_route_as_wps = predict_route_as_wps
        self.hidden_size = hidden_size

        if predict_route_as_wps:
            self.future_waypoints = 20
            self.query_embeds_wps = nn.Parameter(0.02 * torch.randn((1, self.future_waypoints, hidden_size)))
            
            self.route_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim*2), nn.SiLU(True),
                nn.Linear(mlp_dim*2, mlp_dim), nn.SiLU(True), 
                nn.Linear(mlp_dim, 2, bias=False)
            )
            self.route_head.eval()
            
            self.queries = {'route': self.query_embeds_wps}
            self.sizes = {'route': self.future_waypoints}
            self.heads["route"] = self.route_head
            self.order.append('route')

        if speed_wps_mode == '2d':
            dim = 2
        elif speed_wps_mode == '1d':
            dim = 1
        else:
            raise ValueError(f"speed_wps_mode must be '1d' or '2d', not {speed_wps_mode}")
        self.dim = dim
        
        self.future_speed_waypoints = 10
        self.query_embeds_speed = nn.Parameter(0.02 * torch.randn((1, self.future_speed_waypoints, hidden_size)))
        
        self.speed_wps_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim), nn.SiLU(True), 
            nn.Linear(mlp_dim, dim, bias=False)
        )
        self.speed_wps_head.eval()
        
        self.heads["speed_wps"] = self.speed_wps_head
        self.queries['speed_wps'] = self.query_embeds_speed
        self.sizes['speed_wps'] = self.future_speed_waypoints
        self.order.append('speed_wps')

        self.rl_mode = rl_mode
        self.use_user_profile = use_user_profile
        if self.rl_mode:
            if self.use_user_profile:
                self.query_embeds_mode = nn.Parameter(0.02 * torch.randn((1, 1, hidden_size)))
                self.sizes['mode'] = 1
                self.dreaming_head = DreamingActionDiscreteHead(hidden_size=hidden_size, mlp_dim=256, initial_temperature=2.0, use_user_profile=use_user_profile)
            else:
                self.query_embeds_mode = nn.Parameter(0.02 * torch.randn((1, 1, hidden_size)))
                self.sizes['mode'] = 1
                self.dreaming_head = DreamingActionDiscreteHead(hidden_size=hidden_size, mlp_dim=128)
            self.queries['mode'] = self.query_embeds_mode
            self.order.append('mode')

    def forward(self, 
            driving_example: DrivingExample,
            **kwargs
            ) -> Dict[str, Tensor]:

        try:
            driving_input = driving_example.driving_input
        except AttributeError:
            driving_input = driving_example
        
        b = driving_input.camera_images.shape[0]
        inputs = None   
        
        for input_type in self.order:
            query_embed = self.queries[input_type]
            if inputs is None:
                inputs = query_embed.expand(b, -1, -1)
            else:
                inputs = torch.cat((inputs, query_embed.expand(b, -1, -1)), dim=1)
        inputs_mask = torch.ones_like(inputs[:, :, 0], dtype=torch.bool)

        return {"inputs": inputs, "inputs_mask": inputs_mask}

    def get_predictions(
        self, 
        features: Tensor,
        logits: Optional[Tensor] = None,
        rl_mode: bool = False,
        deterministic: bool = True,
    ) -> Dict:

        current_index = 0
        predictions = {}
        

        features_by_type = {}

        self.rl_mode = rl_mode

        for i, input_type in enumerate(self.order):
            size = self.sizes[input_type]
            feature_slice = features[:, current_index: current_index + size]
            
            features_by_type[input_type] = feature_slice

            if input_type == 'mode':
                continue

            prediction = self.heads[input_type](feature_slice).cumsum(1)
            predictions[input_type] = prediction
            
            current_index += size

        if self.rl_mode:
            mode_features = features_by_type.get('mode')
            if mode_features is None:
                return predictions

            if deterministic:
                dreaming_output = self.dreaming_head(mode_features, temperature=1.0)
            else:
                dreaming_output = self.dreaming_head(mode_features)

            dreaming_dist = dreaming_output["dreaming_dist"]

            if deterministic:
                action_idx = dreaming_dist.mode
                actions = self.dreaming_head.decode_indices(action_idx)
            else:
                action_idx = dreaming_dist.sample()
                actions = self.dreaming_head.decode_indices(action_idx)

            action_log_probs = dreaming_dist.log_prob(action_idx)

            predictions["dreaming_actions"] = actions
            predictions["dreaming_action_log_prob"] = action_log_probs
            predictions["dreaming_action_entropy"] = dreaming_dist.entropy()

        current_index += size

        return predictions

class LanguageAdaptor(nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.embed_tokens = language_model.model.embed_tokens
        if hasattr(language_model.model, "lm_head"):
            self.lm_head = language_model.model.lm_head
        elif hasattr(language_model.model, "embed_out"):
            self.lm_head = language_model.model.embed_out
        elif hasattr(language_model.model.base_model.model, 'output'):
            self.lm_head = language_model.model.base_model.model.output
        else:
            raise ValueError("Language model must have `lm_head` or `embed_out` attribute.")


    def forward(self, example: DrivingExample, inference=False, **kwargs) -> Dict[str, Tensor]:
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example
            
        b = driving_input.camera_images.size(0)
        
        if inference:
            label = driving_input.prompt_inference
        else:
            label = driving_input.prompt
            
        if label is not None:
            ids = label.phrase_ids.long()
            ids_valid = label.phrase_valid  # true => is fed into model
            ids_mask = label.loss_masking # true => takes part in loss

        inputs = self.embed_tokens(ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1))
        return {"inputs": inputs, "inputs_mask": ids_valid, "_ids": ids, "_ids_mask": ids_mask}

    def compute_loss(
        self, adaptor_features: Tensor, adaptor_logits: Tensor, inputs: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        del example

        if adaptor_logits is None:
            adaptor_logits = self.lm_head(outputs[:, :-1])
        else:
            adaptor_logits = adaptor_logits[:, :-1]
        labels = torch.where(inputs["_ids_mask"], inputs["_ids"], -1)
        # Shift by 1 for next token prediction
        labels = labels[:, 1:]
        language_loss = F.cross_entropy(
            adaptor_logits.flatten(0, -2), labels.flatten(), ignore_index=-1, reduction="none"
        ).view_as(labels)
        return {"language_loss": (language_loss, labels.ne(-1))}

class UserProfileAdaptor(nn.Module):
    def __init__(self, user_model_path: str, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.database = load_all_user_data(os.environ.get('USER_PROFILE_DATA_DIR', 'user_profile_data/personal_info'), os.environ.get('DATABASE_DIR', 'database/'))

        self.user_profile_encoder = DeclaredProfileEncoder(output_dim=hidden_size)
        self.user_profile_encoder.load_state_dict(torch.load(user_model_path))
        self.user_profile_encoder.eval()

        for param in self.user_profile_encoder.parameters():
            param.requires_grad = False
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,  example: DrivingExample, **kwargs) -> Dict[str, torch.Tensor]:

        with torch.no_grad():
            driver_embeddings = []
            for driver_id in example.driver_id:
                driver_embeddings.append(self.user_profile_encoder(self.database.get(driver_id)['personal_info']))
            driver_embeddings = torch.stack(driver_embeddings, dim=0)

        inputs = driver_embeddings.squeeze(1)
        mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        return {"inputs": inputs, "inputs_mask": mask}

class AdaptorList(nn.Module):
    """
    Each adaptor is responsible for converting a driving example
    to a sequence of tokens and computing the loss on the token outputs.
    Adaptors are only used during training.
    """

    def __init__(
        self,
        driving: Optional[DrivingAdaptor] = None,
        language: Optional[LanguageAdaptor] = None,
        user_profile: Optional[UserProfileAdaptor] = None,
    ):
        super().__init__()
        self.driving = driving
        self.language = language
        self.user_profile = user_profile

        self.language.eval()
        

    @property
    def adaptors(self):
        dct: Dict[str, Adaptor] = {}
        if self.language is not None:
            dct["language"] = self.language
        if self.user_profile is not None:
            dct["user_profile"] = self.user_profile
        if self.driving is not None:
            dct["driving"] = self.driving
        return dct

    def forward(self, example: DrivingExample, **kwargs) -> Dict[str, Tensor]:
        """
        Construct input embeddings for the given driving example.
        """

        input_dict: Dict[str, Tensor] = {}
        inputs_list: List[Tensor] = []
        inputs_mask_list: List[Tensor] = []

        for key, adaptor in self.adaptors.items():
            adaptor_input_dict = adaptor.forward(example, **kwargs)

            
            inputs_list.append(adaptor_input_dict["inputs"])
            inputs_mask_list.append(adaptor_input_dict["inputs_mask"])
            input_dict.update({key + "_" + k: v for k, v in adaptor_input_dict.items()})

        inputs = torch.cat(inputs_list, dim=1)
        inputs_mask = torch.cat(inputs_mask_list, dim=1)
        split_sizes = torch.as_tensor([x.size(1) for x in inputs_list]) # num_of_embeddings for each adaptor
        arange = torch.arange(inputs.size(0), device=inputs.device)[:, None] # [0, 1, 2, ..., B]

        # Apply random permutation of modalities during training
        rand_perm = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), -1) # [B, num_of_embeddings]
        # Apply permutation to move invalid tokens to end of sequence
        valid_perm = inputs_mask[arange, rand_perm].byte().argsort(dim=-1, descending=True, stable=True)
        perm = rand_perm.gather(1, valid_perm)

        input_dict["inputs"] = inputs[arange, perm]
        input_dict["inputs_mask"] = inputs_mask[arange, perm]
        input_dict["perm"] = perm
        input_dict["split_sizes"] = split_sizes
        return input_dict

    def compute_loss(
        self, features: Tensor, logits: Tensor, input_dict: Dict[str, Tensor], example: DrivingExample
    ) -> Dict[str, Tuple[Tensor, Tensor]]:
        """
        Distributes the output embeddings from the transformer to
        the correct loss function and returns a dictionary of losses.
        """

        features_by_adaptor = self.split_outputs_by_adaptor(input_dict, features)
        logits_by_adaptor = self.split_outputs_by_adaptor(input_dict, logits)

        loss_dict: Dict[str, Tuple[Tensor, Tensor]] = {}
        for key, adaptor in self.adaptors.items():
            if key == 'language':
                continue
            adaptor_input_dict = _gather_from_dict(input_dict, key + "_")
            adaptor_features = features_by_adaptor[key]
            adaptor_logits = logits_by_adaptor[key]
            losses = adaptor.compute_loss(adaptor_features, adaptor_logits, adaptor_input_dict, example)
            loss_dict.update(losses)

        return loss_dict

    def split_outputs_by_adaptor(self, input_dict: Dict[str, Tensor], outputs: Tensor) -> Dict[str, Tensor]:
        """
        Splits the output tensor into the correct output for each adaptor, according to the
        split_sizes in the input_dict.
        """
        # First reverse permutation
        inv_perm = input_dict["perm"].argsort(-1)
        arange = torch.arange(inv_perm.size(0), device=inv_perm.device)[:, None]
        outputs = outputs[arange, inv_perm]

        # Now split output for each adaptor
        split_sizes = [int(x) for x in input_dict["split_sizes"]]
        outputs_list = list(outputs.split(split_sizes, dim=1))
        return {key: outputs_list[i] for i, key in enumerate(self.adaptors.keys())}


def _gather_from_dict(d: Dict[str, Tensor], prefix: str):
    out: Dict[str, Tensor] = {}
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out