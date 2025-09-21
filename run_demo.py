#!/usr/bin/env python3
"""
Agricrawler Demo Launcher
Quick setup and run script for judges
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['cv2', 'numpy', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'PIL':
                from PIL import Image
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("📦 Installing demo requirements...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'demo_requirements.txt'])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_demo():
    """Run the webcam demo"""
    print("🚀 Starting Agricrawler Demo...")
    try:
        subprocess.run([sys.executable, 'simple_webcam_demo.py'])
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    """Main demo launcher"""
    print("🌱 AGRICRAWLER DEMO LAUNCHER")
    print("=" * 50)
    print("Setting up demo for judges...")
    print()
    
    # Check if we're in the right directory
    if not Path('simple_webcam_demo.py').exists():
        print("❌ Error: simple_webcam_demo.py not found!")
        print("Please run this script from the hackrobo-main directory")
        return
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"⚠️ Missing packages: {', '.join(missing)}")
        print("Installing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements")
            print("Please install manually: pip install opencv-python numpy Pillow")
            return
    else:
        print("✅ All requirements satisfied!")
    
    print()
    print("🎥 Starting webcam demo...")
    print("Press Ctrl+C to stop the demo")
    print()
    
    # Run the demo
    run_demo()

if __name__ == "__main__":
    main()




