import numpy as np

def euclidean_cost(tracks, detections, track_indices=None,
             detection_indices=None, second_match = False):
    """An euclidean distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        bbox_center = tracks[track_idx].pos
        for column, detection_idx in enumerate(detection_indices):
            candidate = detections[detection_idx].to_xyah()[:2]
            cost_matrix[row, column] = ((candidate[1]-bbox_center[1])**2)+((candidate[0]-bbox_center[0])**2)
    return cost_matrix