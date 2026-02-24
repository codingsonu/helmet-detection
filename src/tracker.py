import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class KalmanBoxTracker:
    """
    Kalman Filter based tracker for bounding boxes
    """
    count = 0
    
    def __init__(self, bbox):
        """
        Initialize tracker with bounding box
        bbox: [x1, y1, x2, y2]
        """
        # Simple state: [x_center, y_center, width, height, vx, vy]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        # Convert bbox to center format
        self.bbox = bbox
        x1, y1, x2, y2 = bbox[:4]
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.w = x2 - x1
        self.h = y2 - y1
        
        # Velocity
        self.vx = 0
        self.vy = 0
        
        # Tracking state
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        
        # History
        self.history = []
    
    def update(self, bbox):
        """Update tracker with new detection"""
        x1, y1, x2, y2 = bbox[:4]
        new_x = (x1 + x2) / 2
        new_y = (y1 + y2) / 2
        
        # Update velocity
        self.vx = new_x - self.x
        self.vy = new_y - self.y
        
        # Update position
        self.x = new_x
        self.y = new_y
        self.w = x2 - x1
        self.h = y2 - y1
        self.bbox = bbox
        
        # Update state
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
    def predict(self):
        """Predict next position"""
        # Simple prediction using velocity
        self.x += self.vx
        self.y += self.vy
        
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Return predicted bbox
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        x2 = self.x + self.w / 2
        y2 = self.y + self.h / 2
        
        return [x1, y1, x2, y2]
    
    def get_state(self):
        """Get current bounding box"""
        x1 = self.x - self.w / 2
        y1 = self.y - self.h / 2
        x2 = self.x + self.w / 2
        y2 = self.y + self.h / 2
        
        return [int(x1), int(y1), int(x2), int(y2)]


class SimpleTracker:
    """
    Simple SORT-like tracker for object tracking
    """
    
    def __init__(self, config):
        """
        Initialize tracker with config
        
        config should have:
            - tracking.max_age: Maximum frames to keep track without update
            - tracking.min_hits: Minimum hits before track is confirmed
            - tracking.iou_threshold: IOU threshold for matching
        """
        self.max_age = config['tracking']['max_age']
        self.min_hits = config['tracking']['min_hits']
        self.iou_threshold = config['tracking']['iou_threshold']
        
        self.trackers = []
        self.frame_count = 0
        
        # Reset tracker ID counter
        KalmanBoxTracker.count = 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: numpy array of shape (N, 5) with [x1, y1, x2, y2, confidence]
                       or list of detections
        
        Returns:
            List of tracked objects with IDs: [{'id': int, 'bbox': [x1,y1,x2,y2]}, ...]
        """
        self.frame_count += 1
        
        # Convert to numpy array if needed
        if isinstance(detections, list):
            if len(detections) == 0:
                detections = np.empty((0, 5))
            else:
                detections = np.array(detections)
        
        if len(detections) == 0:
            detections = np.empty((0, 5))
        
        # Predict new locations of existing trackers
        predicted_boxes = []
        to_delete = []
        
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            predicted_boxes.append(pos)
            
            # Check for invalid predictions
            if np.any(np.isnan(pos)):
                to_delete.append(i)
        
        # Remove invalid trackers
        for i in reversed(to_delete):
            self.trackers.pop(i)
            if i < len(predicted_boxes):
                predicted_boxes.pop(i)
        
        # Match detections to existing trackers
        if len(self.trackers) > 0 and len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                detections, predicted_boxes
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.trackers)))
        
        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx])
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(detections[det_idx])
            self.trackers.append(trk)
        
        # Build output
        results = []
        for trk in self.trackers:
            # Only return tracks that have been confirmed
            if trk.time_since_update < 1:
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    bbox = trk.get_state()
                    results.append({
                        'id': trk.id,
                        'bbox': bbox
                    })
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return results
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """
        Associate detections to tracked objects using IOU
        
        Returns:
            matched: list of (detection_idx, tracker_idx) pairs
            unmatched_detections: list of unmatched detection indices
            unmatched_trackers: list of unmatched tracker indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk)
        
        # Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            # Use linear sum assignment (minimize cost, so we negate IOU)
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = list(zip(row_indices, col_indices))
        else:
            matched_indices = []
        
        # Filter out low IOU matches
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))
        
        for d, t in matched_indices:
            if iou_matrix[d, t] >= self.iou_threshold:
                matched.append((d, t))
                if d in unmatched_detections:
                    unmatched_detections.remove(d)
                if t in unmatched_trackers:
                    unmatched_trackers.remove(t)
        
        return matched, unmatched_detections, unmatched_trackers
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """
        Calculate Intersection over Union (IOU) between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
        
        Returns:
            IOU value (0-1)
        """
        # Ensure we have 4 values
        bbox1 = bbox1[:4]
        bbox2 = bbox2[:4]
        
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Check for no intersection
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def reset(self):
        """Reset tracker state"""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0