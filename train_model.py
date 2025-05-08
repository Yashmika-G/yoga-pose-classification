import os
import argparse
from model_trainer import ModelTrainer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Yoga Pose Detection Model")
    parser.add_argument("--dataset", type=str, default="DATASET",
                        help="Path to the dataset directory")
    parser.add_argument("--model", type=str, default="yoga_model.h5",
                        help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory '{args.dataset}' not found.")
        return
    
    # Check if dataset has the expected structure
    train_dir = os.path.join(args.dataset, "TRAIN")
    test_dir = os.path.join(args.dataset, "TEST")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Dataset should contain 'TRAIN' and 'TEST' directories.")
        return
    
    # Create model trainer
    trainer = ModelTrainer(
        dataset_path=args.dataset,
        model_save_path=args.model
    )
    
    # Process dataset
    print(f"Processing dataset from '{args.dataset}'...")
    if trainer.process_dataset():
        # Train model
        print(f"Training model with {args.epochs} epochs and batch size {args.batch_size}...")
        if trainer.train_model(epochs=args.epochs, batch_size=args.batch_size):
            # Evaluate model
            print("Evaluating model on test set...")
            accuracy = trainer.evaluate_model()
            print(f"Model training complete. Saved to '{args.model}'")
            if accuracy:
                print(f"Test accuracy: {accuracy*100:.2f}%")
        else:
            print("Model training failed.")
    else:
        print("Dataset processing failed.")

if __name__ == "__main__":
    main() 