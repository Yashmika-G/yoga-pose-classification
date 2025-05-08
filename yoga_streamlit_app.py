import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
from pose_detector import PoseDetector
from pose_analyzer import PoseAnalyzer

# Page config
st.set_page_config(
    page_title="Yoga Pose Assistant",
    page_icon="🧘‍♀️",
    layout="wide"
)

class StreamlitYogaApp:
    def __init__(self):
        st.title("Yoga Pose Assistant 🧘‍♀️")
        
        # Sidebar for settings
        with st.sidebar:
            st.header("Settings")
            self.model_path = st.text_input("Model Path", value="yoga_model.h5")
            self.confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8, 0.05)
            self.camera_id = st.number_input("Camera ID", min_value=0, value=1, step=1, 
                                            help="0 for built-in webcam, 1+ for external cameras")
            self.show_landmarks = st.checkbox("Show Pose Landmarks", value=True)
            self.show_guidance = st.checkbox("Show Pose Guidance", value=True)
            
            st.divider()
            st.subheader("Supported Poses")
            st.markdown("- Warrior II (Virabhadrasana II)")
            st.markdown("- Tree Pose (Vrksasana)")
            st.markdown("- Plank Pose")
            st.markdown("- Goddess Pose (Utkata Konasana)")
            st.markdown("- Downward Facing Dog (Adho Mukha Svanasana)")
        
        # Initialize detectors
        if os.path.exists(self.model_path):
            self.pose_detector = PoseDetector(self.model_path)
            st.success(f"Model loaded from: {self.model_path}")
        else:
            self.pose_detector = PoseDetector()
            st.warning(f"Model file not found at: {self.model_path}. Running in limited mode.")
        
        self.pose_analyzer = PoseAnalyzer()
        
        # App state
        self.current_pose = None
        self.pose_confidence = 0.0
        self.pose_score = 0.0
        self.pose_feedback = []
        self.guidance_steps = []
        
        # UI components
        self.col1, self.col2 = st.columns([2, 1])
        
        # Camera frame placeholder
        with self.col1:
            self.frame_placeholder = st.empty()
            self.start_button = st.button("Start Camera", use_container_width=True)
            self.stop_button = st.button("Stop Camera", use_container_width=True)
        
        # Info and guidance
        with self.col2:
            self.pose_info_placeholder = st.empty()
            self.feedback_placeholder = st.empty()
            self.guidance_placeholder = st.empty()
            
    def start_camera(self):
        """Start the camera and process frames"""
        cap = cv2.VideoCapture(int(self.camera_id))
        
        if not cap.isOpened():
            st.error(f"Error: Could not open camera with ID {self.camera_id}")
            return
        
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"Camera started: {frame_width}x{frame_height} at {fps} FPS")
        
        # For FPS calculation
        frame_count = 0
        start_time = time.time()
        fps_display = 0
        
        try:
            while True:
                # Read a frame from the camera
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to capture frame")
                    break
                
                # Flip the frame horizontally for a more intuitive mirror view
                frame = cv2.flip(frame, 1)
                
                # Create a copy of the frame for display
                display_frame = frame.copy()
                
                # Process the frame
                self.process_frame(frame, display_frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1:
                    fps_display = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Add FPS to frame
                cv2.putText(display_frame, f"FPS: {fps_display:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert to RGB for display in Streamlit
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Break the loop if the stop button is pressed
                if self.stop_button:
                    break
                    
                # Add a small delay to reduce CPU usage
                time.sleep(0.01)
                
        finally:
            cap.release()
            self.frame_placeholder.empty()
            st.info("Camera stopped")
    
    def process_frame(self, frame, display_frame):
        """Process a single frame to detect and analyze yoga pose"""
        # Detect pose
        results, _ = self.pose_detector.detect_pose(frame)
        
        # Extract keypoints
        keypoints = self.pose_detector.extract_keypoints(results)
        
        # Draw pose landmarks if enabled
        if self.show_landmarks and results.pose_landmarks:
            display_frame = self.pose_detector.draw_pose_landmarks(display_frame, results)
        
        # If we have valid keypoints, classify the pose
        if np.sum(keypoints) > 0:
            pose_name, confidence = self.pose_detector.classify_pose(keypoints)
            
            # Check if the person is actually doing a yoga pose or just standing
            is_neutral_pose = self._is_neutral_pose(results.pose_landmarks)
            
            # Update current pose if confidence is above threshold and not in neutral pose
            if confidence > self.confidence_threshold and not is_neutral_pose:
                if pose_name != self.current_pose:
                    self.current_pose = pose_name
                    self.guidance_steps = self.pose_analyzer.get_guidance(pose_name)
                
                self.pose_confidence = confidence
                
                # Analyze pose accuracy
                if results.pose_landmarks:
                    pose_desc, score, feedback = self.pose_analyzer.analyze_pose(
                        pose_name, results.pose_landmarks
                    )
                    self.pose_score = score
                    self.pose_feedback = feedback
                
                # Update info displays
                self.update_pose_info()
                self.update_feedback()
                if self.show_guidance:
                    self.update_guidance()
            else:
                # If confidence is low or in neutral pose, show "No pose detected"
                self.current_pose = None
                self.pose_info_placeholder.info("No yoga pose detected. Please try a supported pose.")
                self.feedback_placeholder.empty()
                self.guidance_placeholder.empty()
        else:
            # If no pose detected, display a message
            self.pose_info_placeholder.info("No person detected. Please stand in frame.")
            self.feedback_placeholder.empty()
            self.guidance_placeholder.empty()
    
    def update_pose_info(self):
        """Update the pose information display"""
        if self.current_pose:
            # Create a markdown string with pose info
            pose_info = f"""
            ### Pose Information
            
            **Detected Pose:** {self.current_pose.capitalize()}
            
            **Confidence:** {self.pose_confidence*100:.1f}%
            
            **Accuracy:** {self.pose_score:.1f}%
            """
            
            # Use color-coded containers based on accuracy
            if self.pose_score >= 80:
                self.pose_info_placeholder.success(pose_info)
            elif self.pose_score >= 50:
                self.pose_info_placeholder.warning(pose_info)
            else:
                self.pose_info_placeholder.error(pose_info)
    
    def update_feedback(self):
        """Update the feedback display"""
        if self.pose_feedback:
            feedback_text = "### Improvement Feedback\n\n"
            for item in self.pose_feedback:
                feedback_text += f"- {item}\n"
            self.feedback_placeholder.info(feedback_text)
        else:
            self.feedback_placeholder.empty()
    
    def update_guidance(self):
        """Update the pose guidance display"""
        if self.guidance_steps:
            guidance_text = "### Step-by-Step Guidance\n\n"
            for i, step in enumerate(self.guidance_steps):
                guidance_text += f"{i+1}. {step}\n\n"
            self.guidance_placeholder.info(guidance_text)
        else:
            self.guidance_placeholder.empty()
    
    def _is_neutral_pose(self, landmarks):
        """
        Check if the person is in a neutral standing position rather than a yoga pose
        """
        if not landmarks:
            return True
            
        landmarks_array = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
        
        # Get key body points
        left_shoulder = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_knee = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        left_ankle = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_wrist = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks_array[self.pose_detector.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Check if standing straight (vertical alignment of shoulders, hips, knees, ankles)
        vertical_alignment = True
        
        # Check if legs are straight and together
        left_leg_angle = self._calculate_angle(left_hip[:2], left_knee[:2], left_ankle[:2])
        right_leg_angle = self._calculate_angle(right_hip[:2], right_knee[:2], right_ankle[:2])
        legs_straight = (left_leg_angle > 160 and right_leg_angle > 160)
        
        # Check if arms are down by the sides
        left_arm_down = left_wrist[1] > left_shoulder[1]
        right_arm_down = right_wrist[1] > right_shoulder[1]
        arms_down = left_arm_down and right_arm_down
        
        # The person is in a neutral pose if all conditions are met
        return vertical_alignment and legs_straight and arms_down
        
    def _calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points (in degrees)
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)
        
        return angle

# Main function
def main():
    app = StreamlitYogaApp()
    
    # Start camera if button is pressed
    if app.start_button:
        app.start_camera()

if __name__ == "__main__":
    main() 