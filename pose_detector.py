import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

class PoseDetector:
    def __init__(self, model_path=None):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load or create the pose classification model
        self.model = None
        self.label_encoder = LabelEncoder()
        self.pose_classes = ['warrior2', 'tree', 'plank', 'goddess', 'downdog']
        self.label_encoder.fit(self.pose_classes)
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("No model found. You'll need to train one first.")
    
    def detect_pose(self, image):
        """
        Detect pose landmarks from an image using MediaPipe
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect poses
        results = self.pose.process(image_rgb)
        
        return results, image
    
    def draw_pose_landmarks(self, image, results):
        """
        Draw the pose landmarks on the image
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def extract_keypoints(self, results):
        """
        Extract pose keypoints from MediaPipe results
        """
        if results.pose_landmarks:
            # Extract pose landmarks
            pose = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                            for landmark in results.pose_landmarks.landmark]).flatten()
            return pose
        return np.zeros(33*4)  # Return zeros if no pose detected (33 landmarks with x,y,z,vis)
    
    def classify_pose(self, keypoints):
        """
        Classify the pose using the trained model
        """
        if self.model is None:
            return None, 0.0
        
        # Reshape keypoints for model input
        keypoints = np.expand_dims(keypoints, axis=0)
        
        # Predict pose
        prediction = self.model.predict(keypoints, verbose=0)[0]
        pose_idx = np.argmax(prediction)
        confidence = prediction[pose_idx]
        pose_name = self.label_encoder.inverse_transform([pose_idx])[0]
        
        return pose_name, confidence
    
    def build_and_train_model(self, train_data, train_labels, epochs=20):
        """
        Build and train a pose classification model
        """
        # Encode the labels
        encoded_labels = self.label_encoder.transform(train_labels)
        
        # Convert to categorical
        categorical_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=len(self.pose_classes))
        
        # Build model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(33*4,), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(self.pose_classes), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            train_data,
            categorical_labels,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2
        )
        
        self.model = model
        return model 