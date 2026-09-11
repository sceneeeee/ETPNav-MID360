"""G-Min V0: deterministic coverage of boolean obstacle-admissible cells.

This proposal rule does not certify free space or robot motion. Its only
inputs are a radial-prefix admissibility mask and the candidate budget.
"""
import numpy as np


def circular_sectors(flags):
    """Label connected true bins on a 120-bin circle; false bins get -1."""
    flags = np.asarray(flags)
    if flags.shape != (120,) or flags.dtype != np.bool_:
        raise ValueError("angular flags must be boolean with shape (120,)")
    labels = np.full(120, -1, dtype=np.int64)
    if flags.all():
        return np.zeros(120, dtype=np.int64)
    starts = np.flatnonzero(flags & ~np.roll(flags, 1))
    for label, start in enumerate(starts):
        angle = int(start)
        while flags[angle]:
            labels[angle] = label
            angle = (angle + 1) % 120
    return labels


def _circular_distance(angle, others):
    difference = np.abs(angle - np.asarray(others))
    return np.minimum(difference, 120 - difference)


def select_gmin_candidates(admissibility, max_candidates=5):
    """Return exact cells sorted by (clockwise angle index, radial index).

    Selection priorities are unrepresented sector, angular separation,
    sector interior distance, then smaller angle. Radius never wraps.
    """
    mask = np.asarray(admissibility)
    if mask.shape != (120, 12) or mask.dtype != np.bool_:
        raise ValueError("admissibility must be boolean with shape (120, 12)")
    if (isinstance(max_candidates, (bool, np.bool_))
            or not isinstance(max_candidates, (int, np.integer))
            or not 1 <= max_candidates <= 5):
        raise ValueError("max_candidates must be an integer in 1..5")
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("admissibility violates radial-prefix consistency")
    if not mask.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    extent = int(np.nonzero(mask)[1].max()) + 1
    schedule = ([extent - 1] if max_candidates == 1 else [
        (extent - 1) * (max_candidates - 1 - j) // (max_candidates - 1)
        for j in range(max_candidates)
    ])
    selected = []
    for radius in schedule:
        flags = mask[:, radius]
        sectors = circular_sectors(flags)
        previous_angles = [cell[0] for cell in selected]
        represented = {int(sectors[a]) for a in previous_angles if flags[a]}
        false_angles = np.flatnonzero(~flags)
        eligible = [int(a) for a in np.flatnonzero(flags)
                    if (int(a), radius) not in selected]
        if not eligible:
            continue

        def priority(angle):
            separation = (int(_circular_distance(angle, previous_angles).min())
                          if previous_angles else 0)
            interior = (int(_circular_distance(angle, false_angles).min())
                        if false_angles.size else 0)
            return (int(sectors[angle]) not in represented,
                    separation, interior, -angle)

        selected.append((max(eligible, key=priority), radius))
    cells = np.asarray(sorted(selected), dtype=np.int64).reshape(-1, 2)
    return cells[:, 0], cells[:, 1]
