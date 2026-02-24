import pandas as pd
import cv2
import os
from datetime import datetime

class AnalyticsManager:
    """Manage analytics and logging"""
    
    def __init__(self, config):
        self.config = config
        self.csv_path = config['analytics']['csv_path']
        self.snapshot_path = config['analytics']['snapshot_path']
        self.logs = []
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(self.snapshot_path, exist_ok=True)
    
    def log_frame(self, frame_number, compliance_results):
        """Log compliance data"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for result in compliance_results:
            log_entry = {
                'timestamp': timestamp,
                'frame_number': frame_number,
                'person_id': result['id'],
                'status': result['status'],
                'bbox_x1': result['bbox'][0],
                'bbox_y1': result['bbox'][1],
                'bbox_x2': result['bbox'][2],
                'bbox_y2': result['bbox'][3]
            }
            self.logs.append(log_entry)
    
    def save_violation_snapshot(self, frame, person_id, frame_number, bbox):
        """Save violation snapshot"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"violation_ID{person_id}_Frame{frame_number}_{timestamp}.jpg"
        filepath = os.path.join(self.snapshot_path, filename)
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        crop = frame[max(0, y1-20):y2+20, max(0, x1-20):x2+20]
        cv2.imwrite(filepath, crop)
        
        return filename
    
    def save_csv(self):
        """Save logs to CSV"""
        df = pd.DataFrame(self.logs)
        df.to_csv(self.csv_path, index=False)
        print(f"✓ CSV saved: {self.csv_path}")
    
    def generate_summary(self):
        """Generate summary"""
        df = pd.DataFrame(self.logs)
        
        if df.empty:
            print("No data to summarize")
            return
        
        total = len(df)
        compliant = len(df[df['status'] == 'Compliant'])
        violations = len(df[df['status'] == 'Non-Compliant'])
        
        print("\n=== SUMMARY ===")
        print(f"Total Detections: {total}")
        print(f"Compliant: {compliant}")
        print(f"Violations: {violations}")
        print(f"Compliance Rate: {compliant/total*100:.1f}%")