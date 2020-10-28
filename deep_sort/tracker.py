# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from . import r2_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, max_iou_distance=0.85, max_age=100, n_init=3):
        self.max_iou_distance = max_iou_distance
        self.max_euclidean_distance = 0
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 1st matching
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        #unmatched tracks mean[:2] and pos correction
        translation, translation_pos = [0,0]
        count=0
        for track_idx, detection_idx in matches:
            if self.tracks[track_idx].time_since_update == 1:
                count+=1
                translation += detections[detection_idx].to_xyah()[:2]-self.tracks[track_idx].mean[:2]
                translation_pos += detections[detection_idx].to_xyah()[:2] -self.tracks[track_idx].pos
        if count!=0:
            translation= translation/count
            translation_pos = translation_pos/count
            self.max_euclidean_distance = translation_pos[0] ** 2 + translation_pos[1] ** 2

            for track_idx in unmatched_tracks:
                self.tracks[track_idx].mean[:2] +=translation
                self.tracks[track_idx].pos += translation_pos
        # 2nd matching using new pos 'corrected'
        matches2, unmatched_tracks2, unmatched_detections2 = \
            self._match(detections, track_indices=unmatched_tracks, detection_indices=unmatched_detections, second_match=True)

        # fusionner les matchs
        matches = matches + matches2
        unmatched_tracks = unmatched_tracks2
        unmatched_detections = unmatched_detections2

        #3rd matching
        matches2, unmatched_tracks2, unmatched_detections2 = \
            self._match(detections, track_indices=unmatched_tracks, detection_indices=unmatched_detections,
                        second_match=False, third_match=True)

        # fusionner les matchs
        matches = matches + matches2
        unmatched_tracks = unmatched_tracks2
        unmatched_detections = unmatched_detections2

        # Update track set.
        for track_idx, detection_idx in matches:
            self._next_id = self.tracks[track_idx].update(
                self.kf, detections[detection_idx], self._next_id)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections, track_indices=None, detection_indices=None, second_match=False, third_match=False):

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        #iou_track_candidates = [
        #    i for i, t in enumerate(self.tracks) if t.time_since_update < self.memory]
        if third_match:
            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    r2_matching.euclidean_cost, self.max_euclidean_distance, self.tracks,
                    detections, track_indices, detection_indices, second_match)
        else:
            matches, unmatched_tracks, unmatched_detections = \
                linear_assignment.min_cost_matching(
                    iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                    detections, track_indices, detection_indices, second_match)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        confidence = detection.confidence
        self.tracks.append(Track(
            mean, covariance, self.n_init, self.max_age, confidence, class_name))
        #self._next_id += 1
