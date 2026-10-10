"""Frame-index windows, shared by conversion, verification and loaders."""


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def history_indices(future_start, history_frames=9):
    _positive(future_start, "future_start")
    _positive(history_frames, "history_frames")
    if (history_frames - 1) % 4:
        raise ValueError("history_frames must be 4n+1 (1, 5, 9, ...)")
    if future_start < history_frames:
        return [0] * (history_frames - future_start) + list(range(future_start))
    if history_frames == 1:
        return [future_start - 1]
    return [0] + list(range(future_start - history_frames + 1, future_start))


def window_starts(frame_count, horizon=48, stride=8):
    for value, name in ((frame_count, "frame_count"), (horizon, "horizon"), (stride, "stride")):
        _positive(value, name)
    if horizon % 4:
        raise ValueError("horizon must be a multiple of four for Wan VAE")
    if stride > horizon:
        raise ValueError("stride must be <= horizon to cover every future frame")
    if frame_count < horizon + 1:
        raise ValueError(f"Need at least {horizon + 1} real frames, got {frame_count}; no synthetic future padding")
    last = frame_count - horizon
    starts = list(range(1, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts
