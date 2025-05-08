import cv2
import argparse
import mediapipe as mp

def test_camera(camera_id):
    # Initialize mediapipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return False
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Camera opened successfully: {frame_width}x{frame_height} at {fps} FPS")
    print("Testing pose detection...")
    print("Press 'ESC' to exit, 'S' to toggle skeleton display")
    
    show_skeleton = True
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks if detected and enabled
        if show_skeleton and results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Display status
        status_text = "Pose detection: " + ("SUCCESS" if results.pose_landmarks else "FAILED")
        status_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, status_color, 2)
        
        cv2.putText(frame, "Press 'S' to toggle skeleton display", (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow("Camera Test", frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('s'):  # Toggle skeleton display
            show_skeleton = not show_skeleton
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Camera and Pose Detection")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera device ID (default: 0)")
    
    args = parser.parse_args()
    
    # Test camera
    success = test_camera(args.camera)
    
    if success:
        print("Camera test completed successfully.")
    else:
        print("Camera test failed.")

if __name__ == "__main__":
    main() 