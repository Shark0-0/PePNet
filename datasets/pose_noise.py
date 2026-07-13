import numpy as np


def _cfg_get(cfg, name, default=None):
    return getattr(cfg, name, default) if cfg is not None else default


def apply_pose_noise(pose, cfg=None, split="test", seed=0):
    """Apply deterministic pose corruption for robustness evaluation."""
    if cfg is None or not _cfg_get(cfg, "enabled", False):
        return pose

    apply_to = _cfg_get(cfg, "apply_to", "test")
    if apply_to != "both" and apply_to != split:
        return pose

    out = pose.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)

    gaussian_std = float(_cfg_get(cfg, "gaussian_std", 0.0))
    if gaussian_std > 0:
        if bool(_cfg_get(cfg, "relative_to_body", True)):
            body_span = np.max(out, axis=(0, 1)) - np.min(out, axis=(0, 1))
            scale = max(float(np.linalg.norm(body_span)), 1e-6)
        else:
            scale = 1.0
        noise = rng.normal(0.0, gaussian_std * scale, size=out.shape)
        out = out + noise.astype(np.float32)

    joint_dropout = float(_cfg_get(cfg, "joint_dropout", 0.0))
    if joint_dropout > 0:
        mask = rng.random(out.shape[:2]) < joint_dropout
        out[mask] = 0.0

    return out
