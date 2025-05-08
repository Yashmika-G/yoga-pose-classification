# Yoga Pose Assistant

A real-time yoga pose detection and guidance system that uses computer vision to:
- Detect and identify yoga poses
- Provide feedback on pose accuracy
- Offer step-by-step guidance for improving poses

## Features

- Real-time pose detection using webcam
- Support for 5 yoga poses: warrior2, tree, plank, goddess, downdog
- Detailed feedback on pose alignment and accuracy
- Step-by-step guidance for correcting poses
- Recording capabilities for yoga sessions
- Customizable display options

## System Requirements

- Python 3.8+
- Webcam
- GPU recommended for better performance

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd yoga-pose-assistant
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Testing Your Camera

Before using the main application, test your camera and pose detection capabilities:

```
python test_camera.py
```

Press 'S' to toggle skeleton display and 'ESC' to exit.

## Training the Model

The system comes with a dataset of 5 yoga poses. To train the model:

```
python train_model.py --dataset DATASET --epochs 30
```

Options:
- `--dataset`: Path to the dataset directory (default: DATASET)
- `--model`: Path to save the trained model (default: yoga_model.h5)
- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Training batch size (default: 32)

## Running the Yoga Assistant

Once you have trained the model, you can run the yoga assistant:

```
python run_yoga_assistant.py
```

Options:
- `--model`: Path to the trained model file (default: yoga_model.h5)
- `--threshold`: Confidence threshold for pose detection (default: 0.7)

## Using the Yoga Assistant

When running the assistant, you'll see the camera feed with pose detection and feedback:

### Keyboard Controls:
- `ESC`: Exit the application
- `L`: Toggle pose landmarks display
- `G`: Toggle step-by-step guidance display
- `R`: Toggle recording (saved to the 'recordings' directory)

### UI Components:
1. **Pose Information**: Displays the detected pose, confidence level, and accuracy score
2. **Feedback**: Provides specific feedback to improve your pose
3. **Guidance**: When toggled on, shows step-by-step instructions for the current pose

## Supported Yoga Poses

The system currently supports the following poses:

1. **Warrior II (Virabhadrasana II)**
   - A standing pose that strengthens legs and opens hips

2. **Tree Pose (Vrksasana)**
   - A balancing pose that improves focus and stability

3. **Plank Pose**
   - A core strengthening pose that builds upper body strength

4. **Goddess Pose (Utkata Konasana)**
   - A standing squat that strengthens legs and opens hips

5. **Downward Facing Dog (Adho Mukha Svanasana)**
   - An inverted V-shape pose that stretches the entire body

## Dataset Structure

The dataset is organized as follows:
```
DATASET/
  ├── TRAIN/
  │   ├── warrior2/
  │   ├── tree/
  │   ├── plank/
  │   ├── goddess/
  │   └── downdog/
  └── TEST/
      ├── warrior2/
      ├── tree/
      ├── plank/
      ├── goddess/
      └── downdog/
```

## How It Works

1. **Pose Detection**: Using MediaPipe's pose estimation to detect body landmarks
2. **Pose Classification**: A trained model identifies which yoga pose you're performing
3. **Pose Analysis**: Compares your pose against ideal alignment references
4. **Feedback Generation**: Provides specific guidance based on the analysis

## Extending the System

To add support for additional yoga poses:
1. Add a new folder with images for the pose in both TRAIN and TEST directories
2. Add reference angles and guidance in the `pose_analyzer.py` file
3. Retrain the model using `train_model.py`

## Troubleshooting

- **Camera not working**: Try changing the camera ID with `--camera 1` when running `test_camera.py`
- **Poor detection**: Ensure good lighting and wear contrasting clothes to your background
- **Low performance**: Try reducing the camera resolution in `yoga_assistant.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 