import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os

from src.detector import HelmetDetector
from src.tracker import SimpleTracker
from src.compliance import ComplianceChecker
from src.analytics import AnalyticsManager
from utils.visualization import Visualizer

class HelmetComplianceSystem:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        os.makedirs("output/videos", exist_ok=True)
        os.makedirs("output/logs", exist_ok=True)
        os.makedirs("output/snapshots", exist_ok=True)
        
        # Initialize components
        print("Initializing Helmet Compliance Detection System...")
        self.detector = HelmetDetector(self.config)
        self.tracker = SimpleTracker(self.config)
        self.compliance_checker = ComplianceChecker(self.config)
        self.analytics = AnalyticsManager(self.config)
        self.visualizer = Visualizer(self.config)
        
        print("✓ All components initialized successfully")
    
    def process_video(self, input_path=None, output_path=None):
        """Main video processing pipeline"""
        
        # Use config paths if not provided
        if input_path is None:
            input_path = self.config['video']['input_path']
        if output_path is None:
            output_path = self.config['video']['output_path']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30  # Default FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\nProcessing video: {input_path}")
        print(f"Resolution: {width}x{height} | FPS: {fps} | Frames: {total_frames}")
        
        frame_count = 0
        
        # Progress bar
        pbar = tqdm(total=total_frames, desc="Processing")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if configured
                skip_frames = self.config['video'].get('skip_frames', 1)
                if frame_count % skip_frames != 0:
                    out.write(frame)  # Write original frame
                    pbar.update(1)
                    continue
                
                # Step 1: Detect helmets
                detections = self.detector.detect(frame)
                
                # Filter helmet detections
                helmet_detections = [d for d in detections if d['class_name'].lower() == 'helmet']
                
                # Step 2: Track helmets
                if helmet_detections:
                    helmet_bboxes = np.array([d['bbox'] + [d['confidence']] for d in helmet_detections])
                    tracked_helmets = self.tracker.update(helmet_bboxes)
                else:
                    tracked_helmets = self.tracker.update(np.empty((0, 5)))
                
                # Step 3: Create compliance results
                compliance_results = []
                for tracked in tracked_helmets:
                    compliance_results.append({
                        'id': tracked['id'],
                        'bbox': tracked['bbox'],
                        'status': 'Compliant',
                        'has_helmet': True
                    })
                
                # Step 4: Log analytics
                if compliance_results:
                    self.analytics.log_frame(frame_count, compliance_results)
                
                # Step 5: Visualize
                annotated_frame = self.visualizer.draw_compliance(frame, compliance_results)
                
                # Calculate frame statistics
                stats = {
                    'total': len(compliance_results),
                    'compliant': len(compliance_results),
                    'violations': 0
                }
                annotated_frame = self.visualizer.draw_statistics(annotated_frame, stats)
                
                # Write frame
                out.write(annotated_frame)
                
                pbar.update(1)
        
        except Exception as e:
            print(f"Error during processing: {e}")
            raise
        
        finally:
            # Always release resources
            cap.release()
            out.release()
            pbar.close()
        
        print(f"\n✓ Video processing complete")
        print(f"✓ Output saved to: {output_path}")
        
        # Save analytics
        self.analytics.save_csv()
        self.analytics.generate_summary()
    
    def process_image(self, image_path, output_path=None):
        """Process single image"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Detect
        detections = self.detector.detect(frame)
        helmet_detections = [d for d in detections if d['class_name'].lower() == 'helmet']
        
        # Create compliance results
        compliance_results = [
            {'id': i, 'bbox': d['bbox'], 'status': 'Compliant', 'has_helmet': True} 
            for i, d in enumerate(helmet_detections)
        ]
        
        # Visualize
        annotated_frame = self.visualizer.draw_compliance(frame, compliance_results)
        
        # Save
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_annotated{ext}"
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        cv2.imwrite(output_path, annotated_frame)
        
        print(f"✓ Image processed: {output_path}")
        
        return compliance_results


def main():
    parser = argparse.ArgumentParser(description='Helmet Compliance Detection System')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--input', type=str, help='Input video/image path')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--mode', type=str, default='video', choices=['video', 'image'])
    
    args = parser.parse_args()
    
    # Initialize system
    system = HelmetComplianceSystem(args.config)
    
    # Process
    if args.mode == 'video':
        system.process_video(args.input, args.output)
    else:
        system.process_image(args.input, args.output)


if __name__ == "__main__":
    main()