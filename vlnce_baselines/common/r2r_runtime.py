"""Shared runtime and evaluation-environment setup for formal R2R runs."""

from copy import deepcopy

from vlnce_baselines.common.env_utils import is_slurm_batch_job
from vlnce_baselines.utils import get_camera_orientations12


def initialize_runtime_config(config, local_rank):
    """Attach launcher-owned runtime state before a trainer is constructed."""
    config.defrost()
    config.local_rank = local_rank
    config.freeze()
    return config


def prepare_r2r_evaluation_environment(config):
    """Apply the formal evaluator's pre-environment R2R sensor configuration.

    This deliberately stops before CUDA/device and distributed initialization.
    It is therefore suitable for the policy-free sensor smoke as well as the
    normal evaluator, while keeping both on the identical sensor definition.
    """
    config.defrost()
    task_config = config.TASK_CONFIG
    task_config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    task_config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    task_config.DATASET.ROLES = ["guide"]
    task_config.TASK.MEASUREMENTS = ["POSITION", "STEPS_TAKEN", "COLLISIONS"]
    config.SIMULATOR_GPU_IDS = [config.SIMULATOR_GPU_IDS[config.local_rank]]

    if "HIGHTOLOW" in task_config.TASK.POSSIBLE_ACTIONS:
        index = task_config.TASK.POSSIBLE_ACTIONS.index("HIGHTOLOW")
        task_config.TASK.POSSIBLE_ACTIONS[index] = "HIGHTOLOWEVAL"
    task_config.DATASET.LANGUAGES = config.EVAL.LANGUAGES
    task_config.DATASET.SPLIT = config.EVAL.SPLIT
    task_config.TASK.NDTW.SPLIT = config.EVAL.SPLIT
    task_config.TASK.SDTW.SPLIT = config.EVAL.SPLIT
    config.use_pbar = not is_slurm_batch_job()

    resize_config = config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
    crop_config = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
    for sensor_type in ["RGB", "DEPTH"]:
        resizer_size = dict(resize_config)[sensor_type.lower()]
        cropper_size = dict(crop_config)[sensor_type.lower()]
        sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
        for action, orientation in get_camera_orientations12().items():
            camera_template = f"{sensor_type}_{action}"
            camera_config = deepcopy(sensor)
            camera_config.ORIENTATION = orientation
            camera_config.UUID = camera_template.lower()
            setattr(task_config.SIMULATOR, camera_template, camera_config)
            task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
            resize_config.append((camera_template.lower(), resizer_size))
            crop_config.append((camera_template.lower(), cropper_size))

    config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
    config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
    config.TASK_CONFIG = task_config
    config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
    config.freeze()
    return config
