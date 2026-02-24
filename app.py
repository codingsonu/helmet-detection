import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
import yaml
import pandas as pd
from PIL import Image
import time
import shutil

# Page configuration
st.set_page_config(
    page_title="Helmet Detection Dashboard",
    page_icon="🦺",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'system' not in st.session_state:
    st.session_state.system = None

@st.cache_resource
def load_system():
    """Load helmet detection system"""
    from main import HelmetComplianceSystem
    return HelmetComplianceSystem('config.yaml')

def process_video(video_file, confidence_threshold):
    """Process uploaded video and return results"""
    
    # Create directories
    input_dir = Path("data/input")
    output_dir = Path("output/videos")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded video to input directory (not temp)
    video_path = input_dir / "uploaded_video.mp4"
    output_path = output_dir / "streamlit_result.mp4"
    
    # Write uploaded file
    with open(video_path, 'wb') as f:
        f.write(video_file.getbuffer())
    
    # Load system and update config
    system = load_system()
    system.config['model']['confidence_threshold'] = confidence_threshold
    
    # Clear previous logs
    system.analytics.logs = []
    
    # Process video
    try:
        system.process_video(str(video_path), str(output_path))
        return str(output_path)
    
    except Exception as e:
        st.error(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_analytics():
    """Get analytics from CSV"""
    
    csv_path = "output/logs/compliance_log.csv"
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return None
        
        # Calculate metrics
        total_frames = df['frame_number'].nunique() if 'frame_number' in df.columns else len(df)
        total_detections = len(df)
        unique_persons = df['person_id'].nunique() if 'person_id' in df.columns else len(df)
        
        # Compliance stats
        if 'status' in df.columns:
            compliant = len(df[df['status'] == 'Compliant'])
            non_compliant = len(df[df['status'] == 'Non-Compliant'])
        else:
            compliant = total_detections
            non_compliant = 0
        
        accuracy = (compliant / total_detections * 100) if total_detections > 0 else 0
        
        analytics = {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'unique_persons': unique_persons,
            'compliant': compliant,
            'non_compliant': non_compliant,
            'accuracy': accuracy,
            'dataframe': df
        }
        
        return analytics
    
    except Exception as e:
        st.error(f"Error reading analytics: {e}")
        return None

def main():
    # Header
    st.title("🦺 Helmet Detection Dashboard")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📹 Video Upload", "📊 Accuracy & Results"])
    
    # TAB 1: Video Upload
    with tab1:
        st.header("Upload Video for Helmet Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ Settings")
            confidence = st.slider(
                "Detection Confidence",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Lower = more detections, Higher = more accurate"
            )
            
            st.info("""
            **Instructions:**
            1. Upload video file
            2. Adjust confidence if needed
            3. Click 'Process Video'
            4. Check 'Accuracy & Results' tab
            """)
        
        with col1:
            # File uploader
            video_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload video in MP4, AVI, or MOV format"
            )
            
            if video_file is not None:
                # Display uploaded video
                st.video(video_file)
                
                # Process button
                if st.button("🚀 Process Video", type="primary", use_container_width=True):
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("⏳ Saving video...")
                    progress_bar.progress(10)
                    
                    status_text.text("⏳ Loading AI model...")
                    progress_bar.progress(20)
                    
                    status_text.text("⏳ Processing video... This may take a few minutes")
                    progress_bar.progress(30)
                    
                    # Process video
                    output_path = process_video(video_file, confidence)
                    
                    progress_bar.progress(90)
                    
                    if output_path and os.path.exists(output_path):
                        progress_bar.progress(100)
                        status_text.text("✅ Processing Complete!")
                        
                        st.session_state.processed = True
                        st.session_state.output_video = output_path
                        
                        st.success("✅ Video processed successfully! Go to 'Accuracy & Results' tab")
                        st.balloons()
                        
                        time.sleep(2)
                        status_text.empty()
                        progress_bar.empty()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ Failed to process video. Please try again.")
            
            else:
                st.markdown("""
                    <div style="text-align:center; padding:2rem; border:2px dashed #ccc; border-radius:10px;">
                        <h3>👆 Upload a video file to get started</h3>
                        <p>Supported formats: MP4, AVI, MOV, MKV</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # TAB 2: Accuracy & Results
    with tab2:
        st.header("📊 Detection Results & Accuracy")
        
        if st.session_state.processed and 'output_video' in st.session_state:
            
            # Get analytics
            analytics = get_analytics()
            
            if analytics:
                # Top metrics
                st.subheader("📈 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🎯 Total Detections",
                        value=analytics['total_detections']
                    )
                
                with col2:
                    st.metric(
                        label="👷 Unique Workers",
                        value=analytics['unique_persons']
                    )
                
                with col3:
                    st.metric(
                        label="✅ Compliant",
                        value=analytics['compliant']
                    )
                
                with col4:
                    st.metric(
                        label="📊 Accuracy",
                        value=f"{analytics['accuracy']:.1f}%"
                    )
                
                st.markdown("---")
                
                # Video and stats side by side
                col_video, col_stats = st.columns([3, 2])
                
                with col_video:
                    st.subheader("🎥 Processed Video")
                    if os.path.exists(st.session_state.output_video):
                        st.video(st.session_state.output_video)
                        
                        # Download button
                        with open(st.session_state.output_video, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download Processed Video",
                                data=f,
                                file_name="helmet_detection_result.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                    else:
                        st.warning("Video file not found")
                
                with col_stats:
                    st.subheader("📊 Statistics")
                    
                    # Compliance chart
                    if analytics['compliant'] > 0 or analytics['non_compliant'] > 0:
                        chart_data = pd.DataFrame({
                            'Status': ['Compliant', 'Non-Compliant'],
                            'Count': [analytics['compliant'], analytics['non_compliant']]
                        })
                        st.bar_chart(chart_data.set_index('Status'))
                    
                    # Summary
                    st.markdown("### 📋 Summary")
                    st.write(f"**Total Frames:** {analytics['total_frames']}")
                    st.write(f"**Helmets Detected:** {analytics['compliant']}")
                    st.write(f"**Violations:** {analytics['non_compliant']}")
                    st.write(f"**Compliance Rate:** {analytics['accuracy']:.1f}%")
                
                st.markdown("---")
                
                # Detection log table
                st.subheader("📋 Detection Log")
                
                recent_data = analytics['dataframe'].tail(100)
                st.dataframe(
                    recent_data,
                    use_container_width=True,
                    height=300
                )
                
                # Download CSV
                csv = analytics['dataframe'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Full Report (CSV)",
                    data=csv,
                    file_name="helmet_detection_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Snapshots
                st.markdown("---")
                st.subheader("📸 Detection Snapshots")
                
                snapshot_dir = Path("output/snapshots")
                if snapshot_dir.exists():
                    snapshots = list(snapshot_dir.glob("*.jpg"))
                    
                    if snapshots:
                        cols = st.columns(4)
                        for idx, snapshot in enumerate(snapshots[:8]):
                            with cols[idx % 4]:
                                img = Image.open(snapshot)
                                st.image(img, caption=snapshot.name, use_container_width=True)
                    else:
                        st.info("No snapshots captured")
                else:
                    st.info("No snapshots available")
            
            else:
                st.warning("⚠️ No analytics data found. Try processing the video again.")
        
        else:
            st.info("👈 Please upload and process a video in the 'Video Upload' tab first")
            
            # Sample preview
            st.markdown("---")
            st.subheader("📊 Dashboard Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Detections", "---")
            with col2:
                st.metric("Unique Workers", "---")
            with col3:
                st.metric("Compliant", "---")
            with col4:
                st.metric("Accuracy", "---")

# Footer
def add_footer():
    st.markdown("---")
    st.markdown("""
        <div style="text-align:center; color:gray; padding:10px;">
            <p>🦺 Helmet Detection System | Powered by YOLOv8</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()