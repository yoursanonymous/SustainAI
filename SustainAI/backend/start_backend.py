#!/usr/bin/env python3
"""
Startup script for SustainAI backend
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def train_models():
    """Train the ML models"""
    print("Training ML models...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Models trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training models: {e}")
        return False

def start_server():
    """Start the Flask server"""
    print("Starting Flask server...")
    try:
        subprocess.check_call([sys.executable, "app.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main startup sequence"""
    print("🚀 Starting SustainAI Backend...")
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ Please run this script from the backend directory")
        return
    
    # Install dependencies
    if not install_requirements():
        return
    
    # Train models if they don't exist
    if not os.path.exists("models/nowcast_xgb.pkl"):
        if not train_models():
            return
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()

