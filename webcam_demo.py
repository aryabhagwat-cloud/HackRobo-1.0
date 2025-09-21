#!/usr/bin/env python3
"""
Agricrawler Webcam Demo for Judges
Real-time crop disease detection with bounding boxes and confidence scores
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import time
from pathlib import Path

class PlantDiseaseDetector:
    """Plant disease detection using MobileNetV2"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = [
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
            'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato_healthy', 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'
        ]
        
        # Initialize model (simplified version for demo)
        self.model = self._create_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"🌱 Plant Disease Detector initialized on {self.device}")
        print(f"📊 Supporting {len(self.class_names)} disease classes")
    
    def _create_model(self):
        """Create MobileNetV2 model for disease classification"""
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, len(self.class_names))
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict_disease(self, image):
        """Predict disease from image"""
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Get top 3 predictions
                top3_probs, top3_indices = torch.topk(probabilities, 3)
                
                results = {
                    'predicted_class': self.class_names[predicted.item()],
                    'confidence': confidence.item(),
                    'top3': []
                }
                
                for i in range(3):
                    results['top3'].append({
                        'class': self.class_names[top3_indices[0][i].item()],
                        'confidence': top3_probs[0][i].item()
                    })
                
                return results
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'top3': []
            }

class WebcamDemo:
    """Real-time webcam demo for judges"""
    
    def __init__(self):
        self.detector = PlantDiseaseDetector()
        self.cap = None
        self.demo_running = False
        
        # Demo settings
        self.detection_interval = 1.0  # Detect every 1 second
        self.last_detection_time = 0
        self.current_prediction = None
        
        # Colors for visualization
        self.colors = {
            'healthy': (0, 255, 0),      # Green
            'disease': (0, 0, 255),      # Red
            'text': (255, 255, 255),     # White
            'background': (0, 0, 0)      # Black
        }
    
    def start_demo(self):
        """Start the webcam demo"""
        print("🎥 Starting Agricrawler Webcam Demo")
        print("=" * 50)
        print("Instructions for Judges:")
        print("• Show plant leaves to the camera")
        print("• System will detect diseases in real-time")
        print("• Green box = Healthy plant")
        print("• Red box = Disease detected")
        print("• Press 'q' to quit, 's' to save current frame")
        print("=" * 50)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            return False
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.demo_running = True
        return True
    
    def run_demo(self):
        """Run the main demo loop"""
        if not self.start_demo():
            return
        
        frame_count = 0
        
        try:
            while self.demo_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read from webcam")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Perform detection at intervals
                if current_time - self.last_detection_time >= self.detection_interval:
                    self.current_prediction = self.detector.predict_disease(frame)
                    self.last_detection_time = current_time
                
                # Draw results on frame
                display_frame = self._draw_results(frame)
                
                # Show frame
                cv2.imshow('Agricrawler - Plant Disease Detection Demo', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame)
                elif key == ord('h'):
                    self._show_help()
        
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        
        finally:
            self._cleanup()
    
    def _draw_results(self, frame):
        """Draw detection results on frame"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw bounding box
        box_color = self.colors['healthy'] if 'healthy' in self.current_prediction['predicted_class'].lower() else self.colors['disease']
        cv2.rectangle(display_frame, (50, 50), (width-50, height-50), box_color, 3)
        
        if self.current_prediction:
            # Main prediction
            pred_class = self.current_prediction['predicted_class']
            confidence = self.current_prediction['confidence']
            
            # Clean up class name for display
            display_name = pred_class.replace('___', ' ').replace('__', ' ').replace('_', ' ')
            
            # Draw prediction text
            text = f"{display_name}: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Background for text
            cv2.rectangle(display_frame, (10, 10), (text_size[0] + 20, 50), self.colors['background'], -1)
            cv2.putText(display_frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            # Draw top 3 predictions
            y_offset = 70
            for i, pred in enumerate(self.current_prediction['top3']):
                if i == 0:
                    continue  # Skip first (already shown)
                
                class_name = pred['class'].replace('___', ' ').replace('__', ' ').replace('_', ' ')
                conf_text = f"{i+1}. {class_name}: {pred['confidence']:.2f}"
                
                # Background for additional predictions
                text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(display_frame, (10, y_offset-15), (text_size[0] + 20, y_offset+5), self.colors['background'], -1)
                cv2.putText(display_frame, conf_text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
                y_offset += 25
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save frame", 
            "Press 'h' for help"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (width-200, height-60+i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return display_frame
    
    def _save_frame(self, frame):
        """Save current frame with prediction"""
        if self.current_prediction:
            timestamp = int(time.time())
            filename = f"agricrawler_demo_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Save prediction data
            pred_data = {
                'timestamp': timestamp,
                'prediction': self.current_prediction,
                'filename': filename
            }
            
            with open(f"prediction_{timestamp}.json", 'w') as f:
                json.dump(pred_data, f, indent=2)
            
            print(f"💾 Saved frame and prediction: {filename}")
        else:
            print("⚠️ No prediction available to save")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
🌱 AGRICRAWLER DEMO HELP
========================
This demo shows real-time plant disease detection:

DISEASE CLASSES DETECTED:
• Potato: Early Blight, Late Blight, Healthy
• Tomato: 10 disease types + Healthy
• Pepper: Bacterial Spot + Healthy

COLORS:
• Green box = Healthy plant
• Red box = Disease detected

CONTROLS:
• 'q' = Quit demo
• 's' = Save current frame + prediction
• 'h' = Show this help

TECHNOLOGY:
• MobileNetV2 CNN model
• Real-time inference
• Multi-crop disease detection
• Confidence scoring
        """
        print(help_text)
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Demo cleanup completed")

def main():
    """Main function to run the demo"""
    print("🌱 AGRICRAWLER - Plant Disease Detection Demo")
    print("=" * 60)
    print("For Hackathon Judges")
    print("=" * 60)
    
    # Check if webcam is available
    cap_test = cv2.VideoCapture(0)
    if not cap_test.isOpened():
        print("❌ Error: No webcam detected!")
        print("Please connect a webcam and try again.")
        return
    
    cap_test.release()
    
    # Start demo
    demo = WebcamDemo()
    demo.run_demo()
    
    print("🎉 Demo completed!")
    print("Thank you for viewing the Agricrawler demo!")

if __name__ == "__main__":
    main()




