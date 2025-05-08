import os
import argparse
from yoga_assistant import YogaAssistant

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Yoga Pose Assistant")
    parser.add_argument("--model", type=str, default="yoga_model.h5",
                      help="Path to the trained model file")
    parser.add_argument("--threshold", type=float, default=0.7,
                      help="Confidence threshold for pose detection (0.0-1.0)")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera device ID (default: 0, typically built-in webcam; 1+ for external cameras)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Warning: Model file '{args.model}' not found.")
        print("You need to train a model first using train_model.py")
        print("Proceeding with demonstration mode (limited functionality)")
    
    # Create and start the yoga assistant
    assistant = YogaAssistant(
        model_path=args.model,
        confidence_threshold=args.threshold,
        camera_id=args.camera
    )
    
    print("Starting Yoga Assistant...")
    print("Press 'ESC' to exit")
    print("Press 'L' to toggle pose landmarks")
    print("Press 'G' to toggle guidance")
    print("Press 'R' to toggle recording")
    
    # Start the camera feed
    assistant.start_camera()

if __name__ == "__main__":
    main() 