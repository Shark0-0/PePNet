import torch
import torch.nn.functional as F
from torch import nn

from modules.impstnet import PSTConv
from modules.pose_mamba import PPMamba


class CPS2(nn.Module):
    """Select pose-generated enhancement points with Gumbel-Softmax."""

    def __init__(self, points=512, in_channel=3, mlp1=(64, 128, 256), mlp2=(512,)):
        super().__init__()
        self.points = points

        self.mlp1 = nn.ModuleList()
        for out_channel in mlp1:
            self.mlp1.append(
                nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, 1),
                    nn.BatchNorm1d(out_channel),
                )
            )
            in_channel = out_channel

        self.mlp2 = nn.ModuleList()
        for channel in mlp2:
            self.mlp2.append(
                nn.Sequential(
                    nn.Conv1d(channel, channel, 1),
                    nn.BatchNorm1d(channel),
                )
            )

    def forward(self, x: torch.Tensor, tau):
        _, _, num_points = x.shape
        features = x
        for layer in self.mlp1:
            features = layer(features)
        features = F.relu(features)

        global_feature = torch.max(features, 2)[0]
        batch_size, channels = global_feature.shape
        global_feature = global_feature.view(batch_size, channels, 1).repeat(
            1, 1, num_points
        )
        features = torch.cat([features, global_feature], dim=1)
        for layer in self.mlp2:
            features = layer(features)
        features = F.relu(features)

        probabilities = F.gumbel_softmax(features, hard=False, tau=tau)
        return torch.matmul(probabilities, x[:, :3, :].permute(0, 2, 1))


class imPSTMamba(nn.Module):
    """Pose-enhanced point cloud action recognition network."""

    def __init__(self, config, num_classes):
        super().__init__()
        model_params = config.model.parameters
        radius = model_params.radius
        nsamples = model_params.nsamples

        self.conv1 = PSTConv(
            in_planes=0,
            mid_planes=64,
            out_planes=128,
            spatial_kernel_size=[radius, nsamples],
            temporal_kernel_size=1,
            spatial_stride=4,
            temporal_stride=1,
            temporal_padding=[0, 0],
        )
        self.conv2a = PSTConv(
            in_planes=128,
            mid_planes=256,
            out_planes=512,
            spatial_kernel_size=[2 * radius, nsamples],
            temporal_kernel_size=3,
            spatial_stride=4,
            temporal_stride=1,
            temporal_padding=[0, 0],
        )

        self.mamba = PPMamba(config.model.mamba_parameters)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_params.dim[-1]),
            nn.Linear(model_params.dim[-1], model_params.mlp_dim),
            nn.GELU(),
            nn.Dropout(model_params.dropout2),
            nn.Linear(model_params.mlp_dim, num_classes),
        )

        self.CPS = CPS2(
            config.model.cpsout,
            4,
            config.model.mlp1,
            config.model.mlp2,
        )
        self.pose_edges = config.data.add_points_around_pose.pose_edges
        self.edges_points = config.data.add_points_around_pose.edges_points2
        self.edge_importance_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.edges = len(self.pose_edges)
        self.points_per_edge = self.edges_points
        initial_t = (
            torch.linspace(0, 1, self.points_per_edge + 1)[:-1]
            .repeat(self.edges, 1)
            .unsqueeze(-1)
        )
        self.t_params = nn.Parameter(
            torch.logit(initial_t + 1e-6),
            requires_grad=True,
        )

    def forward(self, input_points, pose, tau, return_features=False):
        batch_size, num_frames, _, _ = input_points.shape
        edges = torch.tensor(self.pose_edges, device=pose.device)
        start_points = pose[..., edges[:, 0], :]
        end_points = pose[..., edges[:, 1], :]

        edge_features = torch.cat([start_points, end_points], dim=-1)
        edge_importance = self.edge_importance_mlp(edge_features)

        interpolation = torch.sigmoid(self.t_params)
        interpolation = interpolation.view(
            1, 1, self.edges, self.points_per_edge, 1
        ).to(pose.device)
        bone_lengths = torch.norm(
            end_points - start_points,
            dim=-1,
            keepdim=True,
        )
        bone_lengths = bone_lengths / bone_lengths.max()
        interpolation = interpolation * bone_lengths.unsqueeze(3)
        interpolation = interpolation / (
            interpolation.max(dim=3, keepdim=True)[0] + 1e-6
        )
        new_points = (
            (1 - interpolation) * start_points.unsqueeze(3)
            + interpolation * end_points.unsqueeze(3)
        )

        total_points = new_points.shape[2] * new_points.shape[3]
        importance = edge_importance.unsqueeze(3).expand(
            -1, -1, -1, self.points_per_edge, -1
        )
        point_features = torch.cat([new_points, importance], dim=-1)
        cps_input = point_features.reshape(
            batch_size * num_frames,
            total_points,
            4,
        ).permute(0, 2, 1)
        sample_points = self.CPS(cps_input, tau)
        sample_points = sample_points.reshape(batch_size, num_frames, -1, 3)

        combined = torch.cat([input_points, sample_points], dim=2)
        new_xyzs, new_features = self.conv1(combined)
        new_features = F.relu(new_features)
        new_xyzs, new_features = self.conv2a(new_xyzs, new_features)
        new_features = F.relu(new_features).permute(0, 1, 3, 2)

        sequence_features = self.mamba(new_xyzs, new_features, pose)
        features = torch.max(sequence_features, dim=1)[0]
        logits = self.mlp_head(features)
        if return_features:
            return logits, features
        return logits
