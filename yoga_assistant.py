import cv2
import numpy as np
import time
import os
from pose_detector import PoseDetector
from pose_analyzer import PoseAnalyzer
import argparse

class YogaAssistant:
    def __init__(self, model_path="yoga_model.h5", confidence_threshold=0.7, mode="assist", camera_id=0):
        # Initialize the pose detector with the trained model
        self.pose_detector = PoseDetector(model_path)
        self.pose_analyzer = PoseAnalyzer()
        self.confidence_threshold = confidence_threshold
        self.mode = mode  # "assist" or "train"
        self.camera_id = camera_id  # Camera device ID
        self.current_pose = None
        self.pose_confidence = 0.0
        self.pose_score = 0.0
        self.pose_feedback = []
        self.show_landmarks = True
        self.show_guidance = False
        self.guidance_steps = []
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)
        self.line_type = 2
        
        # UI colors
        self.colors = {
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255)
        }
        
        # Create output directory for recordings
        self.record = False
        self.out = None
        self.output_dir = "recordings"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def start_camera(self):
        """
        Start the camera feed and process frames
        """
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {self.camera_id}.")
            return
        
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera started: {frame_width}x{frame_height} at {fps} FPS")
        
        # Start time for FPS calculation
        start_time = time.time()
        frame_count = 0
        
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Process the frame
            processed_frame = self.process_frame(frame, display_frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display FPS on the frame
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                        (10, 30), self.font, self.font_scale, self.font_color, self.line_type)
            
            # Add instructions on the frame
            self.draw_instructions(processed_frame)
            
            # Show the frame
            cv2.imshow("Yoga Assistant", processed_frame)
            
            # Record video if enabled
            if self.record and self.out is not None:
                self.out.write(processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == 27:  # ESC key to exit
                break
            elif key == ord('l'):  # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
            elif key == ord('g'):  # Toggle guidance
                self.show_guidance = not self.show_guidance
            elif key == ord('r'):  # Toggle recording
                self.toggle_recording(frame_width, frame_height, fps)
            
        # Release resources
        if self.out is not None:
            self.out.release()
        cap.release()
        cv2.destroyAllWindows()
    
    def process_frame(self, frame, display_frame):
        """
        Process a single frame to detect and analyze yoga pose
        """
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
            
            # Update current pose if confidence is above threshold
            if confidence > self.confidence_threshold:
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
            
            # Draw pose information on the frame
            display_frame = self.draw_pose_info(display_frame)
            
            # Draw guidance if enabled
            if self.show_guidance:
                display_frame = self.draw_guidance(display_frame)
        else:
            # If no pose detected, display a message
            cv2.putText(display_frame, "No pose detected. Please stand in frame.", 
                        (20, 60), self.font, self.font_scale, self.colors["red"], self.line_type)
        
        return display_frame
    
    def draw_pose_info(self, frame):
        """
        Draw pose information on the frame
        """
        if self.current_pose is None:
            return frame
        
        # Draw a semi-transparent overlay at the bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 180), (frame.shape[1], frame.shape[0]), 
                      self.colors["black"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw pose name and confidence
        cv2.putText(frame, f"Pose: {self.current_pose.capitalize()}", 
                    (20, frame.shape[0] - 150), self.font, self.font_scale, 
                    self.colors["white"], self.line_type)
        
        cv2.putText(frame, f"Confidence: {self.pose_confidence*100:.1f}%", 
                    (20, frame.shape[0] - 120), self.font, self.font_scale, 
                    self.colors["white"], self.line_type)
        
        # Draw accuracy score
        score_color = self.colors["green"] if self.pose_score >= 80 else \
                      self.colors["yellow"] if self.pose_score >= 50 else \
                      self.colors["red"]
        
        cv2.putText(frame, f"Accuracy: {self.pose_score:.1f}%", 
                    (20, frame.shape[0] - 90), self.font, self.font_scale, 
                    score_color, self.line_type)
        
        # Draw feedback
        if self.pose_feedback:
            for i, feedback in enumerate(self.pose_feedback[:2]):  # Show at most 2 feedback items
                cv2.putText(frame, f"• {feedback}", 
                            (20, frame.shape[0] - 60 + i*30), self.font, self.font_scale, 
                            self.colors["yellow"], self.line_type)
        
        return frame
    
    def draw_guidance(self, frame):
        """
        Draw step-by-step guidance on the frame
        """
        if not self.guidance_steps:
            return frame
        
        # Draw a semi-transparent overlay on the right side
        overlay = frame.copy()
        guidance_width = 400
        cv2.rectangle(overlay, (frame.shape[1] - guidance_width, 0), 
                      (frame.shape[1], frame.shape[0] - 180), 
                      self.colors["black"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw guidance title
        cv2.putText(frame, "Step-by-Step Guidance:", 
                    (frame.shape[1] - guidance_width + 10, 30), 
                    self.font, self.font_scale, self.colors["white"], self.line_type)
        
        # Draw guidance steps
        for i, step in enumerate(self.guidance_steps):
            # Split long text into multiple lines
            words = step.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + word) < 35:  # Limit line length
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            
            if current_line:
                lines.append(current_line)
            
            # Draw each line
            for j, line in enumerate(lines):
                y_pos = 60 + i*50 + j*25
                if y_pos < frame.shape[0] - 190:  # Ensure text stays within bounds
                    cv2.putText(frame, f"{i+1}. " + line if j == 0 else "   " + line, 
                                (frame.shape[1] - guidance_width + 10, y_pos), 
                                self.font, 0.6, self.colors["white"], 1)
        
        return frame
    
    def draw_instructions(self, frame):
        """
        Draw keyboard instructions on the frame
        """
        instructions = [
            "ESC: Exit",
            "L: Toggle landmarks",
            "G: Toggle guidance",
            "R: Toggle recording"
        ]
        
        # Draw a semi-transparent overlay at the top-right
        overlay = frame.copy()
        instruction_width = 200
        instruction_height = len(instructions) * 30 + 10
        cv2.rectangle(overlay, (frame.shape[1] - instruction_width, 0), 
                      (frame.shape[1], instruction_height), 
                      self.colors["black"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw instructions
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                        (frame.shape[1] - instruction_width + 10, 25 + i*30), 
                        self.font, 0.6, self.colors["white"], 1)
        
        return frame
    
    def toggle_recording(self, width, height, fps):
        """
        Toggle video recording
        """
        if self.record:
            self.record = False
            if self.out is not None:
                self.out.release()
                self.out = None
            print("Recording stopped")
        else:
            self.record = True
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_file = os.path.join(self.output_dir, f"yoga_session_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"Recording started: {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Yoga Pose Assistant")
    parser.add_argument("--model", type=str, default="yoga_model.h5", 
                        help="Path to the trained model file")
    parser.add_argument("--threshold", type=float, default=0.7, 
                        help="Confidence threshold for pose detection")
    parser.add_argument("--mode", type=str, default="assist", choices=["assist", "train"],
                        help="Mode of operation: assist or train")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    
    args = parser.parse_args()
    
    # Create and start the yoga assistant
    assistant = YogaAssistant(
        model_path=args.model,
        confidence_threshold=args.threshold,
        mode=args.mode,
        camera_id=args.camera
    )
    
    print("Starting Yoga Assistant...")
    print("Press 'ESC' to exit, 'L' to toggle landmarks, 'G' to toggle guidance, 'R' to toggle recording")
    
    assistant.start_camera()

if __name__ == "__main__":
    main() 