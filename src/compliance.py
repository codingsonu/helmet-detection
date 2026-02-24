import time
from collections import defaultdict

class ComplianceChecker:
    """Check helmet compliance"""
    
    def __init__(self, config):
        self.config = config
        self.violation_history = defaultdict(list)
        self.violation_cooldown = config['analytics']['violation_cooldown']
    
    def check_compliance(self, person_detections, helmet_detections, tracked_persons):
        """Determine compliance status"""
        compliance_results = []
        
        for helmet in helmet_detections:
            if helmet['class_name'].lower() == 'helmet':
                compliance_results.append({
                    'id': len(compliance_results),
                    'bbox': helmet['bbox'],
                    'status': 'Compliant',
                    'has_helmet': True,
                    'confidence': helmet['confidence'],
                    'timestamp': time.time()
                })
        
        return compliance_results
    
    def should_save_violation(self, person_id, frame_count):
        """Check if violation should be saved"""
        if person_id not in self.violation_history:
            self.violation_history[person_id] = [frame_count]
            return True
        
        last_violation = self.violation_history[person_id][-1]
        if frame_count - last_violation >= self.violation_cooldown:
            self.violation_history[person_id].append(frame_count)
            return True
        
        return False