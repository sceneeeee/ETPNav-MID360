"""Atomic, provenance-locked recovery for a fixed calibration selection."""
import copy
import json
import os
from pathlib import Path
import tempfile

from sparse_depth_feature_alignment import ChannelMoments


class SensorStability:
    """A lucky valid frame is insufficient: require two consecutive exact matches."""
    def __init__(self):
        self.previous = None
        self.stable = False

    def observe(self, statistics, valid):
        current = {k: (v['shape'], v['dtype'], v['sha256']) for k,v in statistics.items()} if valid else None
        self.stable = current is not None and current == self.previous
        self.previous = current
        return self.stable


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
        directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CalibrationProgress:
    """Commit both conditions together; completed IDs must be a selection prefix."""
    def __init__(self, selection, provenance, feature_shape):
        self.selection = copy.deepcopy(selection)
        ids = [r['episode_id'] for r in selection]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('selection requires unique ordered episode IDs')
        self.provenance = copy.deepcopy(provenance)
        self.feature_shape = tuple(feature_shape)
        self.records = []
        self.moments = {name: ChannelMoments() for name in ('dense', 'sparse')}

    @property
    def completed_ids(self):
        return [r['episode_id'] for r in self.records]

    @property
    def next_index(self):
        return len(self.records)

    def add(self, record, features):
        if self.next_index >= len(self.selection):
            raise ValueError('selection already complete')
        expected = self.selection[self.next_index]
        if any(record[k] != expected[k] for k in ('episode_id', 'scene_id')):
            raise ValueError('duplicate, out-of-order, or wrong-scene observation')
        if set(features) != set(self.moments):
            raise ValueError('both feature conditions are required')
        updated = {}
        for name, tensor in features.items():
            if tuple(tensor.shape) != self.feature_shape:
                raise ValueError('feature shape changed during calibration')
            updated[name] = ChannelMoments.from_state(self.moments[name].to_state())
            updated[name].update(tensor)
        self.moments = updated
        self.records.append(copy.deepcopy(record))

    def to_state(self):
        return dict(schema_version=1, selection=self.selection, provenance=self.provenance,
                    feature_shape=list(self.feature_shape), completed_episode_ids=self.completed_ids,
                    next_calibration_index=self.next_index, calibration_index_base=0,
                    records=self.records, moments={k:v.to_state() for k,v in self.moments.items()})

    def save(self, path):
        atomic_json(path, self.to_state())

    @classmethod
    def load(cls, path, selection, provenance, feature_shape):
        state = json.loads(Path(path).read_text())
        if state['schema_version'] != 1 or state['calibration_index_base'] != 0:
            raise ValueError('unsupported recovery schema')
        if state['selection'] != selection:
            raise ValueError('calibration selection changed')
        if state['provenance'] != provenance:
            raise ValueError('calibration provenance changed; refusing resume')
        if tuple(state['feature_shape']) != tuple(feature_shape):
            raise ValueError('calibration feature shape changed')
        result = cls(selection, provenance, feature_shape)
        result.records = state['records']
        completed = result.completed_ids
        if (len(completed) > len(selection) or completed != [r['episode_id'] for r in selection[:len(completed)]] or
                completed != state['completed_episode_ids'] or len(completed) != state['next_calibration_index']):
            raise ValueError('completed IDs must be an exact selection prefix')
        for record, expected in zip(result.records, selection):
            if record['scene_id'] != expected['scene_id']:
                raise ValueError('completed scene differs from selection')
        if set(state['moments']) != {'dense', 'sparse'}:
            raise ValueError('missing calibration condition')
        result.moments = {k:ChannelMoments.from_state(v) for k,v in state['moments'].items()}
        for moment in result.moments.values():
            if moment.observations != len(completed) or (completed and moment.shape != result.feature_shape):
                raise ValueError('moments disagree with completed selection/feature shape')
        return result
