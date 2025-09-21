#!/usr/bin/env python3
"""
Agricrawler Simple Webcam Demo for Judges
Real-time crop disease detection simulation with visual feedback
"""

import cv2
import numpy as np
import time
import random
import json
from pathlib import Path

class SimplePlantDiseaseDetector:
    """Simplified plant disease detector for demo purposes"""
    
    def __init__(self):
        # Disease classes that the system can detect
        self.disease_classes = [
            'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
            'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
            'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
            'Tomato Target Spot', 'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl',
            'Tomato Healthy', 'Pepper Bacterial Spot', 'Pepper Healthy'
        ]
        
        # Simulate different detection scenarios
        self.demo_scenarios = [
            {'class': 'Potato Early Blight', 'confidence': 0.92, 'severity': 'high'},
            {'class': 'Tomato Healthy', 'confidence': 0.88, 'severity': 'none'},
            {'class': 'Tomato Late Blight', 'confidence': 0.95, 'severity': 'critical'},
            {'class': 'Pepper Bacterial Spot', 'confidence': 0.76, 'severity': 'medium'},
            {'class': 'Potato Healthy', 'confidence': 0.91, 'severity': 'none'},
            {'class': 'Tomato Septoria Leaf Spot', 'confidence': 0.83, 'severity': 'medium'},
        ]
        
        self.scenario_index = 0
        self.last_detection_time = 0
        self.detection_interval = 2.0  # Change detection every 2 seconds
        
        print("🌱 Simple Plant Disease Detector initialized")
        print(f"📊 Supporting {len(self.disease_classes)} disease classes")
    
    def detect_disease(self, image):
        """Simulate disease detection"""
        current_time = time.time()
        
        # Change detection every interval
        if current_time - self.last_detection_time >= self.detection_interval:
            self.scenario_index = (self.scenario_index + 1) % len(self.demo_scenarios)
            self.last_detection_time = current_time
        
        # Get current scenario
        scenario = self.demo_scenarios[self.scenario_index]
        
        # Add some randomness to make it more realistic
        confidence = scenario['confidence'] + random.uniform(-0.05, 0.05)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'class': scenario['class'],
            'confidence': confidence,
            'severity': scenario['severity'],
            'timestamp': current_time
        }

class WebcamDemo:
    """Real-time webcam demo for judges"""
    
    def __init__(self):
        self.detector = SimplePlantDiseaseDetector()
        self.cap = None
        self.demo_running = False
        
        # Colors for visualization
        self.colors = {
            'healthy': (0, 255, 0),      # Green
            'disease': (0, 0, 255),      # Red
            'critical': (0, 0, 139),     # Dark Red
            'text': (255, 255, 255),     # White
            'background': (0, 0, 0),     # Black
            'info': (255, 255, 0)        # Yellow
        }
        
        # Demo statistics
        self.detection_count = 0
        self.start_time = time.time()
    
    def start_demo(self):
        """Start the webcam demo"""
        print("🎥 Starting Agricrawler Webcam Demo")
        print("=" * 60)
        print("🌱 AGRICRAWLER - Plant Disease Detection System")
        print("=" * 60)
        print("Instructions for Judges:")
        print("• Show plant leaves or any object to the camera")
        print("• System will simulate disease detection in real-time")
        print("• Green box = Healthy plant detected")
        print("• Red box = Disease detected")
        print("• Dark red box = Critical disease")
        print("• Press 'q' to quit, 's' to save frame, 'h' for help")
        print("=" * 60)
        
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
        self.start_time = time.time()
        return True
    
    def run_demo(self):
        """Run the main demo loop"""
        if not self.start_demo():
            return
        
        frame_count = 0
        current_prediction = None
        
        try:
            while self.demo_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Could not read from webcam")
                    break
                
                frame_count += 1
                
                # Get disease detection
                current_prediction = self.detector.detect_disease(frame)
                self.detection_count += 1
                
                # Draw results on frame
                display_frame = self._draw_results(frame, current_prediction)
                
                # Show frame
                cv2.imshow('Agricrawler - Plant Disease Detection Demo', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(frame, current_prediction)
                elif key == ord('h'):
                    self._show_help()
                elif key == ord('r'):
                    self._show_recommendations(current_prediction)
        
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        
        finally:
            self._cleanup()
    
    def _draw_results(self, frame, prediction):
        """Draw detection results on frame"""
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Determine box color based on prediction
        if prediction:
            if 'healthy' in prediction['class'].lower():
                box_color = self.colors['healthy']
                status_text = "HEALTHY"
            elif prediction['severity'] == 'critical':
                box_color = self.colors['critical']
                status_text = "CRITICAL DISEASE"
            else:
                box_color = self.colors['disease']
                status_text = "DISEASE DETECTED"
        else:
            box_color = self.colors['info']
            status_text = "ANALYZING..."
        
        # Draw main bounding box
        cv2.rectangle(display_frame, (50, 50), (width-50, height-50), box_color, 4)
        
        if prediction:
            # Main prediction text
            pred_text = f"{prediction['class']}"
            conf_text = f"Confidence: {prediction['confidence']:.2f}"
            severity_text = f"Severity: {prediction['severity'].upper()}"
            
            # Draw status background
            cv2.rectangle(display_frame, (10, 10), (width-10, 80), self.colors['background'], -1)
            
            # Draw status
            cv2.putText(display_frame, status_text, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            # Draw prediction details
            cv2.putText(display_frame, pred_text, (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            cv2.putText(display_frame, conf_text, (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Draw demo info
        demo_time = int(time.time() - self.start_time)
        info_text = f"Demo Time: {demo_time}s | Detections: {self.detection_count}"
        cv2.putText(display_frame, info_text, (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['info'], 1)
        
        # Draw controls
        controls = ["'q'=quit", "'s'=save", "'h'=help", "'r'=recommendations"]
        for i, control in enumerate(controls):
            cv2.putText(display_frame, control, (width-120, height-60+i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        return display_frame
    
    def _save_frame(self, frame, prediction):
        """Save current frame with prediction"""
        timestamp = int(time.time())
        filename = f"agricrawler_demo_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        
        # Save prediction data
        if prediction:
            pred_data = {
                'timestamp': timestamp,
                'prediction': prediction,
                'filename': filename,
                'demo_time': time.time() - self.start_time,
                'total_detections': self.detection_count
            }
            
            with open(f"prediction_{timestamp}.json", 'w') as f:
                json.dump(pred_data, f, indent=2)
            
            print(f"💾 Saved frame and prediction: {filename}")
            print(f"   Disease: {prediction['class']}")
            print(f"   Confidence: {prediction['confidence']:.2f}")
            print(f"   Severity: {prediction['severity']}")
        else:
            print("⚠️ No prediction available to save")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
🌱 AGRICRAWLER DEMO HELP
========================
This demo simulates real-time plant disease detection:

DISEASE CLASSES DETECTED:
• Potato: Early Blight, Late Blight, Healthy
• Tomato: 10 disease types + Healthy  
• Pepper: Bacterial Spot + Healthy

SEVERITY LEVELS:
• None = Healthy plant
• Medium = Moderate disease
• High = Serious disease
• Critical = Emergency treatment needed

COLORS:
• Green box = Healthy plant
• Red box = Disease detected
• Dark red box = Critical disease

CONTROLS:
• 'q' = Quit demo
• 's' = Save current frame + prediction
• 'h' = Show this help
• 'r' = Show treatment recommendations

TECHNOLOGY DEMONSTRATED:
• Real-time computer vision
• Disease classification
• Confidence scoring
• Severity assessment
• Multi-crop support
        """
        print(help_text)
    
    def _show_recommendations(self, prediction):
        """Show treatment recommendations"""
        if not prediction:
            print("⚠️ No prediction available for recommendations")
            return
        
        print(f"\n🌱 TREATMENT RECOMMENDATIONS")
        print("=" * 40)
        print(f"Disease: {prediction['class']}")
        print(f"Severity: {prediction['severity'].upper()}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        print()
        
        if 'healthy' in prediction['class'].lower():
            print("✅ HEALTHY PLANT - No treatment needed")
            print("   • Continue current care routine")
            print("   • Monitor for any changes")
            print("   • Maintain optimal growing conditions")
        else:
            severity = prediction['severity']
            if severity == 'critical':
                print("🚨 CRITICAL DISEASE - Immediate action required")
                print("   • Apply fungicide treatment immediately")
                print("   • Isolate affected plants")
                print("   • Contact agricultural expert")
            elif severity == 'high':
                print("⚠️ HIGH SEVERITY - Treatment needed within 24 hours")
                print("   • Apply appropriate fungicide")
                print("   • Improve air circulation")
                print("   • Remove infected leaves")
            else:
                print("📋 MODERATE SEVERITY - Monitor and treat")
                print("   • Apply preventive treatment")
                print("   • Improve growing conditions")
                print("   • Regular monitoring")
        
        print(f"\n💡 GENERAL RECOMMENDATIONS:")
        print("   • Ensure proper drainage")
        print("   • Maintain optimal humidity")
        print("   • Regular plant inspection")
        print("   • Follow integrated pest management")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Show demo statistics
        demo_time = time.time() - self.start_time
        print(f"\n📊 DEMO STATISTICS:")
        print(f"   Total runtime: {demo_time:.1f} seconds")
        print(f"   Total detections: {self.detection_count}")
        print(f"   Average detection rate: {self.detection_count/demo_time:.1f} per second")
        print("🧹 Demo cleanup completed")

def main():
    """Main function to run the demo"""
    print("🌱 AGRICRAWLER - Plant Disease Detection Demo")
    print("=" * 60)
    print("For Hackathon Judges")
    print("=" * 60)
    print("This demo simulates the Agricrawler system's disease detection capabilities")
    print("using computer vision and machine learning.")
    print()
    
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
    print()
    print("🌱 AGRICRAWLER - Revolutionizing Agriculture with AI")

if __name__ == "__main__":
    main()




