import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="SegmentationEnv-v0",
    entry_point="segmentation.envs:BasicEnv",
    kwargs={"objs_dir": "../point_cloud_env/objects",
        "sample_size": 1024, 
        "wall_sample_size": 0.3,
        "unlabelled_punishment": 1.0,
        "false_labels_punishment": 1.0,
        "diff_punishment": 0.025,
        "max_steps_per_scene": 15,
        "debug": False,
        "max_scenes": 4,
        "scene_mode": "Random",
        "training": True,
        "point_mode": "None",
        "voxel_size": 60,
        "voxel_mode": "None",
        "single_scenes": False,
        "early_diff": False,
        "wall_weight": 0.5}
)
