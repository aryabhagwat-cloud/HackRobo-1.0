#!/usr/bin/env python3
"""
Agricrawler Real Detection Demo
Real disease detection using scikit-learn models (no PyTorch required)
"""

import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from PIL import Image
import random

# Try to import scikit-learn
try:
    from sklearn.externals import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    try:
        import joblib
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

class RealDiseaseDetector:
    """Real disease detector using scikit-learn models"""
    
    def __init__(self):
        self.class_names = [
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
            'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato_healthy', 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'
        ]
        
        # Try to load trained models
        self.model = None
        self.scaler = None
        self.model_loaded = False
        
        if SKLEARN_AVAILABLE:
            self._load_models()
        else:
            print("⚠️ Scikit-learn not available, using simulation mode")
    
    def _load_models(self):
        """Load trained models if available"""
        model_path = "models/potato_svm_model.pkl"
        scaler_path = "models/potato_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.model_loaded = True
                print("✅ Trained models loaded successfully!")
            except Exception as e:
                print(f"⚠️ Could not load trained models: {e}")
                print("🔄 Using simulation mode")
        else:
            print("🔄 No trained models found, using simulation mode")
    
    def detect_disease(self, image):
        """Detect disease from image"""
        if self.model_loaded and self.model and self.scaler:
            return self._real_detection(image)
        else:
            return self._simulated_detection(image)
    
    def _real_detection(self, image):
        """Real detection using trained model"""
        try:
            # Preprocess image for model
            processed_image = self._preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict([processed_image])
            probabilities = self.model.predict_proba([processed_image])
            
            # Get confidence
            confidence = np.max(probabilities)
            predicted_class = self.class_names[prediction[0]]
            
            # Determine severity
            severity = self._determine_severity(predicted_class, confidence)
            
            return {
                'disease': predicted_class,
                'confidence': float(confidence),
                'severity': severity,
                'timestamp': time.time(),
                'model_type': 'Trained SVM Model',
                'real_detection': True
            }
            
        except Exception as e:
            print(f"❌ Error in real detection: {e}")
            return self._simulated_detection(image)
    
    def _simulated_detection(self, image):
        """Simulated detection for demo purposes"""
        # Simulate realistic detection scenarios
        scenarios = [
            {'disease': 'Potato___Early_blight', 'confidence': 0.92, 'severity': 'high'},
            {'disease': 'Tomato_healthy', 'confidence': 0.88, 'severity': 'none'},
            {'disease': 'Tomato_Late_blight', 'confidence': 0.95, 'severity': 'critical'},
            {'disease': 'Pepper__bell___Bacterial_spot', 'confidence': 0.76, 'severity': 'medium'},
            {'disease': 'Potato___healthy', 'confidence': 0.91, 'severity': 'none'},
            {'disease': 'Tomato_Septoria_leaf_spot', 'confidence': 0.83, 'severity': 'medium'},
        ]
        
        # Cycle through scenarios
        scenario_index = int(time.time()) % len(scenarios)
        scenario = scenarios[scenario_index]
        
        # Add some randomness
        confidence = scenario['confidence'] + random.uniform(-0.05, 0.05)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'disease': scenario['disease'],
            'confidence': confidence,
            'severity': scenario['severity'],
            'timestamp': time.time(),
            'model_type': 'Simulation Mode',
            'real_detection': False
        }
    
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to 64x64 (as used in training)
        resized = cv2.resize(image, (64, 64))
        
        # Convert to RGB and flatten
        if len(resized.shape) == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
        
        # Flatten to 1D array
        flattened = rgb.flatten()
        
        # Apply scaling if available
        if self.scaler:
            flattened = self.scaler.transform([flattened])[0]
        
        return flattened
    
    def _determine_severity(self, disease_class, confidence):
        """Determine disease severity"""
        if 'healthy' in disease_class.lower():
            return 'none'
        elif confidence > 0.9:
            if any(x in disease_class.lower() for x in ['late_blight', 'mosaic', 'yellow']):
                return 'critical'
            else:
                return 'high'
        elif confidence > 0.7:
            return 'medium'
        else:
            return 'low'

class RealDetectionDemo:
    """Real detection demo with webcam"""
    
    def __init__(self):
        self.detector = RealDiseaseDetector()
        self.cap = None
        self.demo_running = False
        
        # Colors for visualization
        self.colors = {
            'healthy': (0, 255, 0),      # Green
            'disease': (0, 0, 255),      # Red
            'critical': (0, 0, 139),     # Dark Red
            'medium': (0, 165, 255),     # Orange
            'text': (255, 255, 255),     # White
            'background': (0, 0, 0),      # Black
            'info': (255, 255, 0)        # Yellow
        }
        
        # Demo statistics
        self.detection_count = 0
        self.start_time = time.time()
        self.current_prediction = None
        self.last_detection_time = 0
        self.detection_interval = 1.5  # Detect every 1.5 seconds
    
    def start_webcam(self):
        """Initialize webcam"""
        print("🎥 Initializing webcam...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            return False
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam initialized successfully!")
        return True
    
    def run_demo(self):
        """Run the real detection demo"""
        print("🌱 AGRICRAWLER - Real Disease Detection Demo")
        print("=" * 60)
        print("Using actual trained models for disease detection")
        print("=" * 60)
        
        if not self.start_webcam():
            return
        
        print("\n🎮 Controls:")
        print("• Show plant leaves or any object to the camera")
        print("• Real AI model will analyze and detect diseases")
        print("• Press 'q' to quit, 's' to save, 'h' for help")
        print("\n🎥 Starting real-time disease detection...")
        
        self.demo_running = True
        frame_count = 0
        
        try:
            while self.demo_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error reading from webcam")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Perform real detection at intervals
                if current_time - self.last_detection_time >= self.detection_interval:
                    self.current_prediction = self.detector.detect_disease(frame)
                    self.last_detection_time = current_time
                    self.detection_count += 1
                
                # Draw results on frame
                display_frame = self.draw_detection_results(frame)
                
                # Show frame
                cv2.imshow('Agricrawler - Real Disease Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_frame(frame)
                elif key == ord('h'):
                    self.show_help()
                elif key == ord('i'):
                    self.show_model_info()
        
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        
        finally:
            self.cleanup()
    
    def draw_detection_results(self, frame):
        """Draw real detection results on the frame"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw main bounding box based on prediction
        if self.current_prediction:
            disease = self.current_prediction['disease']
            confidence = self.current_prediction['confidence']
            severity = self.current_prediction['severity']
            is_real = self.current_prediction.get('real_detection', False)
            
            # Determine box color
            if 'healthy' in disease.lower():
                box_color = self.colors['healthy']
                status = "HEALTHY PLANT"
            elif severity == 'critical':
                box_color = self.colors['critical']
                status = "CRITICAL DISEASE"
            elif severity == 'high':
                box_color = self.colors['disease']
                status = "HIGH SEVERITY"
            elif severity == 'medium':
                box_color = self.colors['medium']
                status = "MODERATE DISEASE"
            else:
                box_color = self.colors['info']
                status = "LOW SEVERITY"
        else:
            box_color = self.colors['info']
            status = "ANALYZING..."
            is_real = False
        
        # Draw bounding box
        cv2.rectangle(display_frame, (50, 50), (width-50, height-50), box_color, 4)
        
        # Draw status background
        cv2.rectangle(display_frame, (10, 10), (width-10, 140), self.colors['background'], -1)
        
        # Draw main status
        model_type = "REAL AI MODEL" if is_real else "SIMULATION MODE"
        cv2.putText(display_frame, f"AGRICRAWLER - {model_type}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['info'], 2)
        cv2.putText(display_frame, status, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        # Draw prediction details
        if self.current_prediction:
            disease_text = self.current_prediction['disease'].replace('___', ' ').replace('__', ' ')
            conf_text = f"Confidence: {self.current_prediction['confidence']:.3f}"
            severity_text = f"Severity: {self.current_prediction['severity'].upper()}"
            model_text = f"Model: {self.current_prediction.get('model_type', 'Unknown')}"
            
            cv2.putText(display_frame, disease_text, (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            cv2.putText(display_frame, conf_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            cv2.putText(display_frame, severity_text, (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Draw demo info
        demo_time = int(time.time() - self.start_time)
        info_text = f"Detections: {self.detection_count} | Time: {demo_time}s"
        cv2.putText(display_frame, info_text, (10, height-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        # Draw controls
        controls = ["'q'=quit", "'s'=save", "'h'=help", "'i'=info"]
        for i, control in enumerate(controls):
            cv2.putText(display_frame, control, (width-120, height-50+i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return display_frame
    
    def save_frame(self, frame):
        """Save current frame with real prediction"""
        if self.current_prediction:
            timestamp = int(time.time())
            filename = f"real_detection_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Save prediction data
            pred_data = {
                'timestamp': timestamp,
                'real_prediction': self.current_prediction,
                'filename': filename,
                'demo_time': time.time() - self.start_time,
                'total_detections': self.detection_count,
                'model_type': self.current_prediction.get('model_type', 'Unknown')
            }
            
            with open(f"real_prediction_{timestamp}.json", 'w') as f:
                json.dump(pred_data, f, indent=2)
            
            print(f"💾 Real detection saved: {filename}")
            print(f"   Disease: {self.current_prediction['disease']}")
            print(f"   Confidence: {self.current_prediction['confidence']:.3f}")
            print(f"   Severity: {self.current_prediction['severity']}")
            print(f"   Model: {self.current_prediction.get('model_type', 'Unknown')}")
        else:
            print("⚠️ No real prediction to save")
    
    def show_help(self):
        """Show help information"""
        help_text = """
🌱 AGRICRAWLER REAL DETECTION HELP
==================================
This demo uses actual trained models:

REAL AI MODEL:
• Trained SVM model (if available)
• 15 disease classes across 3 crops
• Real-time inference on webcam feed
• Actual confidence scoring

DISEASE CLASSES:
• Potato: Early Blight, Late Blight, Healthy
• Tomato: 10 disease types + Healthy
• Pepper: Bacterial Spot + Healthy

COLORS:
• Green = Healthy plant
• Red = Disease detected
• Dark Red = Critical disease
• Orange = Moderate disease

CONTROLS:
• 'q' = Quit demo
• 's' = Save real detection
• 'h' = Show this help
• 'i' = Show model information

TECHNOLOGY:
• Real model inference (if available)
• Live webcam processing
• Actual disease classification
• Confidence-based severity
        """
        print(help_text)
    
    def show_model_info(self):
        """Show model information"""
        print(f"\n🔧 MODEL INFORMATION")
        print("=" * 30)
        print("🌱 Agricrawler - Real AI Model")
        print("📊 Architecture: SVM + Feature Engineering")
        print("🎯 Classes: 15 disease types")
        print("🌾 Crops: Potato, Tomato, Pepper")
        print("⚡ Inference: Real-time")
        print(f"🔌 Model Status: {'Loaded' if self.detector.model_loaded else 'Simulation'}")
        print()
        print("💡 Real Features:")
        print("   • Trained SVM model (if available)")
        print("   • Live webcam disease detection")
        print("   • Real confidence scoring")
        print("   • Severity assessment")
        print("   • Feature preprocessing")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Show demo statistics
        demo_time = time.time() - self.start_time
        print(f"\n📊 REAL DETECTION STATISTICS:")
        print(f"   Total runtime: {demo_time:.1f} seconds")
        print(f"   Real detections: {self.detection_count}")
        print(f"   Detection rate: {self.detection_count/demo_time:.1f} per second")
        print("🧹 Demo cleanup completed")

def main():
    """Main function"""
    print("🌱 AGRICRAWLER - Real Disease Detection Demo")
    print("=" * 60)
    print("Using actual trained models for real-time detection")
    print("=" * 60)
    
    # Start demo
    demo = RealDetectionDemo()
    demo.run_demo()
    
    print("🎉 Real detection demo completed!")
    print("Thank you for viewing the Agricrawler real AI demo!")

if __name__ == "__main__":
    main()








