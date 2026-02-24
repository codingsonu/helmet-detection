import cv2

class Visualizer:
    """Visualize detections on frames"""
    
    def __init__(self, config):
        self.config = config
        self.compliant_color = tuple(config['visualization']['compliant_color'])
        self.non_compliant_color = tuple(config['visualization']['non_compliant_color'])
        self.bbox_thickness = config['visualization']['bbox_thickness']
        self.font_scale = config['visualization']['font_scale']
    
    def draw_compliance(self, frame, compliance_results):
        """Draw bounding boxes on frame"""
        annotated = frame.copy()
        
        for result in compliance_results:
            bbox = result['bbox']
            person_id = result['id']
            status = result['status']
            
            color = self.compliant_color if status == 'Compliant' else self.non_compliant_color
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            label = f"ID:{person_id} {status}"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, 2)
        
        return annotated
    
    def draw_statistics(self, frame, stats):
        """Draw statistics overlay"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, f"Total: {stats.get('total', 0)}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Compliant: {stats.get('compliant', 0)}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.compliant_color, 2)
        cv2.putText(frame, f"Violations: {stats.get('violations', 0)}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.non_compliant_color, 2)
        
        return frame