import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pose_detector import PoseDetector

class ModelTrainer:
    def __init__(self, dataset_path, model_save_path="yoga_model.h5"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.pose_detector = PoseDetector()
        self.train_data = []
        self.train_labels = []
        self.model = None
        
    def process_dataset(self):
        """
        Process the dataset by extracting pose landmarks from all images
        """
        print("Processing dataset...")
        
        # Get all pose classes (from directory names)
        pose_classes = [d for d in os.listdir(os.path.join(self.dataset_path, "TRAIN")) 
                        if os.path.isdir(os.path.join(self.dataset_path, "TRAIN", d))]
        
        for pose_class in pose_classes:
            print(f"Processing class: {pose_class}")
            pose_dir = os.path.join(self.dataset_path, "TRAIN", pose_class)
            
            # Get all images for this pose
            images = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Process each image
            for img_path in tqdm(images):
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue
                    
                    # Detect pose landmarks
                    results, _ = self.pose_detector.detect_pose(img)
                    
                    # Extract keypoints
                    keypoints = self.pose_detector.extract_keypoints(results)
                    
                    # Add to dataset only if pose was detected
                    if np.sum(keypoints) != 0:
                        self.train_data.append(keypoints)
                        self.train_labels.append(pose_class)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Processed {len(self.train_data)} images for {len(pose_classes)} classes")
        
        # Convert to numpy arrays
        self.train_data = np.array(self.train_data)
        
        return len(self.train_data) > 0
        
    def train_model(self, epochs=20, batch_size=32):
        """
        Train the pose classification model
        """
        if len(self.train_data) == 0:
            print("No training data available. Run process_dataset first.")
            return False
            
        print(f"Training model with {len(self.train_data)} samples...")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.train_data, self.train_labels, test_size=0.2, random_state=42
        )
        
        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)
        
        y_train_encoded = label_encoder.transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        
        # Convert to categorical
        num_classes = len(label_encoder.classes_)
        y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
        y_val_categorical = tf.keras.utils.to_categorical(y_val_encoded, num_classes)
        
        # Build model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train,
            y_train_categorical,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val_categorical),
            verbose=1
        )
        
        # Save model
        model.save(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        # Save class mapping
        class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        np.save("class_mapping.npy", class_mapping)
        
        self.model = model
        
        # Plot training history
        self._plot_training_history(history)
        
        return True
    
    def evaluate_model(self):
        """
        Evaluate the model on the test set
        """
        if self.model is None:
            print("No model available. Train model first.")
            return
            
        # Process test dataset
        test_data = []
        test_labels = []
        
        # Get all pose classes (from directory names)
        test_dir = os.path.join(self.dataset_path, "TEST")
        pose_classes = [d for d in os.listdir(test_dir) 
                        if os.path.isdir(os.path.join(test_dir, d))]
        
        for pose_class in pose_classes:
            print(f"Processing test class: {pose_class}")
            pose_dir = os.path.join(test_dir, pose_class)
            
            # Get all images for this pose
            images = [os.path.join(pose_dir, f) for f in os.listdir(pose_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Process each image
            for img_path in tqdm(images):
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Detect pose landmarks
                    results, _ = self.pose_detector.detect_pose(img)
                    
                    # Extract keypoints
                    keypoints = self.pose_detector.extract_keypoints(results)
                    
                    # Add to dataset only if pose was detected
                    if np.sum(keypoints) != 0:
                        test_data.append(keypoints)
                        test_labels.append(pose_class)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if len(test_data) == 0:
            print("No test data available.")
            return
            
        # Convert to numpy arrays
        test_data = np.array(test_data)
        
        # Encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(self.train_labels)
        
        test_labels_encoded = label_encoder.transform(test_labels)
        
        # Convert to categorical
        num_classes = len(label_encoder.classes_)
        test_labels_categorical = tf.keras.utils.to_categorical(test_labels_encoded, num_classes)
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(test_data, test_labels_categorical)
        
        print(f"Test accuracy: {accuracy*100:.2f}%")
        
        return accuracy
    
    def _plot_training_history(self, history):
        """
        Plot training and validation accuracy and loss
        """
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(dataset_path="DATASET")
    if trainer.process_dataset():
        trainer.train_model(epochs=30)
        trainer.evaluate_model() 