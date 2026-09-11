WAYPOINT_OUTPUT_KEYS = (
    "cand_rgb",
    "cand_depth",
    "cand_angle_fts",
    "cand_img_idxes",
    "cand_angles",
    "cand_distances",
    "pano_rgb",
    "pano_depth",
    "pano_angle_fts",
    "pano_img_idxes",
)


class WaypointOutput(dict):
    """Dict-compatible named output for the unchanged ETP planner contract."""

    def __init__(self, **fields):
        _validate_field_names(fields)
        super().__init__((key, fields[key]) for key in WAYPOINT_OUTPUT_KEYS)

    def __getattr__(self, name):
        if name in WAYPOINT_OUTPUT_KEYS:
            return self[name]
        raise AttributeError(name)


def _validate_field_names(output):
    missing = [key for key in WAYPOINT_OUTPUT_KEYS if key not in output]
    extra = [key for key in output if key not in WAYPOINT_OUTPUT_KEYS]
    if missing or extra:
        raise ValueError(
            "invalid waypoint output fields: missing={}, extra={}".format(
                missing, extra
            )
        )


def validate_waypoint_output(output):
    _validate_field_names(output)
    return output
