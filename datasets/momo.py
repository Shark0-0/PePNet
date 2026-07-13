import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from datasets.pose_noise import apply_pose_noise

# Cross_Subject 和 Cross_Action 列表保持不变
Cross_Subject = [2,3,4,5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
            25,26,27,28,29,30,32,41,45,46,47,48,49,50,51,53,54,55,62,
            70,71,74,75,76,77,78,79,80,81,82,83,84,85,90,92,94,95]

Cross_Action = [1, 2, 4, 6, 8, 9, 11, 13, 15, 16, 17, 19, 20]

class MomoDataset(Dataset):
    def __init__(self, data_config, train=True): # Changed signature
        super(MomoDataset, self).__init__()

        self.data_cfg = data_config # Store the config object
        self.train = train

        self.root = self.data_cfg.path
        self.pose_root = self.data_cfg.pose_path
        self.frames_per_clip = self.data_cfg.clip_len
        self.frame_interval = getattr(self.data_cfg, 'frame_interval', 1) # Use getattr for optional params
        self.num_points = self.data_cfg.num_points
        self.pose_type = self.data_cfg.pose_type # 'spike' or 'hmr'
        self.pose_noise_cfg = getattr(self.data_cfg, "pose_noise", None)

        self.file_paths = []
        self.labels = []
        self.index_map = []
        index = 0

        for file_name in os.listdir(self.root):
            if file_name.endswith('.pkl'):
                person_id = int(file_name[1:4])
                label = int(file_name[9:12]) - 1

                if 0 <= label <= 29: # 确保标签有效
                    if train:
                        if person_id in Cross_Subject:
                            file_path = os.path.join(self.root, file_name)
                            self.file_paths.append(file_path)
                            self.labels.append(label)
                            self.index_map.append(index)
                            index += 1
                    else:
                        if person_id not in Cross_Subject:
                            file_path = os.path.join(self.root, file_name)
                            self.file_paths.append(file_path)
                            self.labels.append(label)
                            self.index_map.append(index)
                            index += 1

        if not self.labels:
            self.num_classes = 30
            print("Warning: No data loaded for this split (train/test). Check Cross_Subject/Action lists and data paths.")
        else:
            self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index = self.index_map[idx]
        label = self.labels[index]

        with open(self.file_paths[index], 'rb') as f:
            raw_data = pickle.load(f) # This is a list of arrays (frames)

        sampled_frame_indices = self._sample_frame_indices(len(raw_data))
        clip_frames = [raw_data[i] for i in sampled_frame_indices] # List of (N_points, 3)

        # Load pose data
        base_file_name = os.path.basename(self.file_paths[index])
        pose_file_path = os.path.join(self.pose_root, base_file_name)

        if not os.path.exists(pose_file_path):
            num_pose_joints = 15 if self.pose_type == 'hmr' else 13 # Example joint counts
            pose_frames_data = [np.zeros((num_pose_joints, 3), dtype=np.float32) for _ in sampled_frame_indices]
        else:
            with open(pose_file_path, 'rb') as f:
                pose_data_raw = pickle.load(f) # List of (N_pose_joints, 3) arrays
            pose_frames_data = [pose_data_raw[i] for i in sampled_frame_indices] # Sample corresponding pose frames

        # Coordinate transformation for 'spike' dataset poses
        if self.pose_type == 'spike' and pose_frames_data and pose_frames_data[0].shape[1] == 3:
             pose_frames_data = [p[:, [2, 0, 1]] if p is not None and p.ndim == 2 else p for p in pose_frames_data]


        processed_clip = []
        processed_poses = []

        previous_sampled_frame_for_empty = np.random.rand(self.num_points, 3).astype(np.float32) # fallback

        for i in range(self.frames_per_clip):
            frame_pc = clip_frames[i]
            pose_joints = pose_frames_data[i] # (N_joints, 3)

            # pose
            if pose_joints is not None and pose_joints.size > 0:
                mean_pose = np.mean(pose_joints, axis=0, keepdims=True)
                current_pose_processed = pose_joints - mean_pose
            else: # Handle case where pose might be missing for a frame
                num_expected_joints = 15 if self.pose_type == 'hmr' else 13
                current_pose_processed = np.zeros((num_expected_joints, 3), dtype=np.float32)

            # Sample points from point cloud
            if frame_pc is None or len(frame_pc) == 0:
                current_frame_processed = previous_sampled_frame_for_empty
            else:
                # Center point cloud
                mean_pc = np.mean(frame_pc, axis=0, keepdims=True)
                frame_pc_centered = frame_pc - mean_pc

                if len(frame_pc_centered) > self.num_points:
                    indices = np.random.choice(frame_pc_centered.shape[0], self.num_points, replace=False)
                    current_frame_processed = frame_pc_centered[indices]
                elif len(frame_pc_centered) < self.num_points:
                    padding = np.zeros((self.num_points - len(frame_pc_centered), frame_pc_centered.shape[1]), dtype=np.float32)
                    current_frame_processed = np.vstack((frame_pc_centered, padding))
                else:
                    current_frame_processed = frame_pc_centered

            processed_clip.append(current_frame_processed.astype(np.float32))
            processed_poses.append(current_pose_processed.astype(np.float32))
            previous_sampled_frame_for_empty = current_frame_processed

        final_clip = np.array(processed_clip) # (T, N_points, 3)
        final_poses = np.array(processed_poses) # (T, N_pose_joints, 3)
        split = "train" if self.train else "test"
        final_poses = apply_pose_noise(
            final_poses,
            cfg=self.pose_noise_cfg,
            split=split,
            seed=int(getattr(self.data_cfg, "seed", 0)) + int(index),
        )

        data = {
            'clip': final_clip,
            'pose': final_poses,
        }
        return data, label, index

    def _sample_frame_indices(self, total_frames):
        """Helper to sample frame indices for a clip."""
        if total_frames == 0: # Should not happen if file_paths are valid
            return [0] * self.frames_per_clip # Return dummy indices

        indices = []
        if total_frames >= self.frames_per_clip:
            if self.train:
                # Randomly sample start of segments
                for i in range(self.frames_per_clip):
                    segment_start = int(total_frames * i / self.frames_per_clip)
                    segment_end = int(total_frames * (i + 1) / self.frames_per_clip)
                    # Ensure segment_end is exclusive for randint upper bound if it's used directly
                    # and handle segment_start == segment_end
                    if segment_start == segment_end : # if total_frames is small
                         indices.append(min(segment_start, total_frames -1))
                    else:
                         indices.append(np.random.randint(segment_start, segment_end))

            else: # Test: sample middle of segments
                for i in range(self.frames_per_clip):
                    segment_start = int(total_frames * i / self.frames_per_clip)
                    segment_end = int(total_frames * (i + 1) / self.frames_per_clip)
                    indices.append(min( (segment_start + segment_end) // 2, total_frames -1) )
        else: # Pad if not enough frames
            indices = list(range(total_frames))
            indices.extend([total_frames - 1] * (self.frames_per_clip - total_frames)) # Pad with last frame

        indices = [min(idx, total_frames - 1) for idx in indices]
        return indices
