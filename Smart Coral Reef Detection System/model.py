import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd


class CoralBleachingDetector:
    def __init__(self, use_ml=False, model_path=None):
        """
        Initialize the coral bleaching detection system.
        
        Args:
            use_ml (bool): Whether to use machine learning for classification
            model_path (str): Path to a pre-trained model (if use_ml is True)
        """
        self.use_ml = use_ml
        self.model = None
        
        # Color ranges for healthy and bleached corals (in HSV)
        # These are initial values and should be calibrated based on specific coral species
        self.healthy_coral_lower = np.array([0, 100, 100])  # More saturated colors
        self.healthy_coral_upper = np.array([180, 255, 255])
        
        self.bleached_coral_lower = np.array([0, 0, 200])   # White/pale colors
        self.bleached_coral_upper = np.array([180, 30, 255])
        
        # Load ML model if requested
        if use_ml and model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        elif use_ml:
            print("ML mode enabled but no model loaded. Using color-based detection as fallback.")
    
    def create_ml_model(self, input_shape=(224, 224, 3)):
        """
        Create a CNN model for coral classification
        
        Args:
            input_shape (tuple): Input image dimensions
            
        Returns:
            model: Compiled TensorFlow model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and fully connected layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')  # 2 classes: healthy and bleached
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, training_data_dir, validation_split=0.2, epochs=20, batch_size=32):
        """
        Train the ML model on coral images
        
        Args:
            training_data_dir (str): Path to directory with 'healthy' and 'bleached' subdirectories
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            history: Training history
        """
        if not self.model:
            self.create_ml_model()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            training_data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation data generator
        validation_generator = train_datagen.flow_from_directory(
            training_data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs
        )
        
        return history
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def preprocess_image(self, image):
        """
        Preprocess image for analysis
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            processed_image: Preprocessed image
        """
        # Convert to HSV for better color segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(hsv_image, (5, 5), 0)
        
        return blurred
    
    def detect_bleaching_color_based(self, image):
        """
        Detect coral bleaching based on color thresholds
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            result_image: Image with highlighted regions
            bleaching_percentage: Estimated percentage of bleaching
        """
        # Make a copy for visualization
        result_image = image.copy()
        
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Create masks for healthy and bleached coral
        healthy_mask = cv2.inRange(processed, self.healthy_coral_lower, self.healthy_coral_upper)
        bleached_mask = cv2.inRange(processed, self.bleached_coral_lower, self.bleached_coral_upper)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        healthy_mask = cv2.morphologyEx(healthy_mask, cv2.MORPH_OPEN, kernel)
        bleached_mask = cv2.morphologyEx(bleached_mask, cv2.MORPH_OPEN, kernel)
        
        # Count pixels in each mask
        total_coral_pixels = cv2.countNonZero(healthy_mask) + cv2.countNonZero(bleached_mask)
        if total_coral_pixels > 0:
            bleaching_percentage = (cv2.countNonZero(bleached_mask) / total_coral_pixels) * 100
        else:
            bleaching_percentage = 0
        
        # Highlight the detected regions in the result image
        result_image[healthy_mask > 0] = [0, 255, 0]  # Green for healthy coral
        result_image[bleached_mask > 0] = [255, 255, 255]  # White for bleached coral
        
        return result_image, bleaching_percentage
    
    def detect_bleaching_ml(self, image):
        """
        Detect coral bleaching using the ML model
        
        Args:
            image: Input image
            
        Returns:
            prediction: Class probabilities [healthy_prob, bleached_prob]
        """
        if not self.model:
            print("ML model not loaded, falling back to color-based detection")
            return self.detect_bleaching_color_based(image)
        
        # Resize and preprocess for the model
        resized = cv2.resize(image, (224, 224))
        normalized = resized / 255.0
        batch = np.expand_dims(normalized, axis=0)
        
        # Get prediction
        prediction = self.model.predict(batch)[0]
        
        # Create a heatmap visualization
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Get class activation map using gradients (simplified)
        # In a real application, you'd implement Grad-CAM or similar
        # This is a simplified placeholder
        if prediction[1] > prediction[0]:  # More likely to be bleached
            # Apply a simple red overlay with transparency proportional to confidence
            overlay = result_image.copy()
            alpha = prediction[1]  # Transparency based on confidence
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha * 0.5, result_image, 1 - alpha * 0.5, 0, result_image)
            
            bleaching_percentage = prediction[1] * 100
        else:
            # Apply a simple green overlay for healthy coral
            overlay = result_image.copy()
            alpha = prediction[0]  # Transparency based on confidence
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha * 0.5, result_image, 1 - alpha * 0.5, 0, result_image)
            
            bleaching_percentage = (1 - prediction[0]) * 100
        
        return result_image, bleaching_percentage
    
    def analyze_image(self, image_path):
        """
        Analyze an image for coral bleaching
        
        Args:
            image_path: Path to input image
            
        Returns:
            result_image: Image with detection visualization
            bleaching_percentage: Estimated percentage of bleaching
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Use ML or color-based detection
        if self.use_ml and self.model:
            result, percentage = self.detect_bleaching_ml(image)
        else:
            result, percentage = self.detect_bleaching_color_based(image)
        
        return result, percentage
    
    def analyze_video(self, video_path, output_path=None, sample_rate=30):
        """
        Analyze a video for coral bleaching
        
        Args:
            video_path: Path to input video
            output_path: Path to save the output video
            sample_rate: Process every Nth frame
            
        Returns:
            time_series: DataFrame with bleaching percentages over time
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video if requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Prepare time series data
        time_points = []
        bleaching_values = []
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process only every Nth frame
            if frame_count % sample_rate == 0:
                # Process frame
                if self.use_ml and self.model:
                    result, percentage = self.detect_bleaching_ml(frame)
                else:
                    result, percentage = self.detect_bleaching_color_based(frame)
                
                # Add timestamp (in seconds) and bleaching percentage
                time_points.append(frame_count / fps)
                bleaching_values.append(percentage)
                
                # Add timestamp and percentage to the frame
                time_str = f"Time: {frame_count/fps:.2f}s"
                cv2.putText(result, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                bleach_str = f"Bleaching: {percentage:.2f}%"
                cv2.putText(result, bleach_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                # Save to output video if requested
                if writer:
                    writer.write(result)
                
                # Display progress
                if frame_count % (sample_rate * 10) == 0:
                    print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        
        # Create time series DataFrame
        time_series = pd.DataFrame({
            'Time (s)': time_points,
            'Bleaching (%)': bleaching_values
        })
        
        return time_series
    
    def analyze_drone_images(self, image_dir, output_dir=None):
        """
        Analyze a series of drone images and generate a reef health map
        
        Args:
            image_dir: Directory containing drone images
            output_dir: Directory to save results
            
        Returns:
            summary_data: DataFrame with summary statistics for each image
        """
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Initialize results storage
        results = []
        
        # Process each image
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            
            try:
                # Analyze image
                result_img, bleaching_pct = self.analyze_image(img_path)
                
                # Save results
                results.append({
                    'Image': img_file,
                    'Bleaching (%)': bleaching_pct
                })
                
                # Save output image if requested
                if output_dir:
                    out_path = os.path.join(output_dir, f"analyzed_{img_file}")
                    cv2.imwrite(out_path, result_img)
                
                print(f"Analyzed {img_file}: {bleaching_pct:.2f}% bleaching")
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Generate summary visualization if we have any results
        if results and output_dir:
            plt.figure(figsize=(12, 6))
            plt.bar(summary_df['Image'], summary_df['Bleaching (%)'])
            plt.xticks(rotation=90)
            plt.title('Coral Bleaching Analysis by Image')
            plt.ylabel('Bleaching Percentage')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bleaching_summary.png'))
        
        return summary_df


# Example usage:
if __name__ == "__main__":
    # Initialize the detector (choose whether to use ML)
    detector = CoralBleachingDetector(use_ml=False)
    
    # Example 1: Analyze a single image
    # result_image, bleaching_pct = detector.analyze_image("sample_coral.jpg")
    # cv2.imshow("Coral Bleaching Analysis", result_image)
    # cv2.waitKey(0)
    # print(f"Estimated bleaching: {bleaching_pct:.2f}%")
    
    # Example 2: Train an ML model (if you have a dataset)
    # detector.create_ml_model()
    # history = detector.train_model("coral_dataset/")
    # detector.save_model("coral_bleaching_model.h5")
    
    # Example 3: Analyze a video 
    # time_series = detector.analyze_video("coral_video.mp4", "output_video.mp4")
    # time_series.to_csv("bleaching_over_time.csv")
    
    # Example 4: Process a batch of drone images
    # summary = detector.analyze_drone_images("drone_images/", "results/")
    # summary.to_csv("bleaching_summary.csv")
    
    print("Coral bleaching detection system ready!")
