# ADAS-lane-Assist-in-Unity3D
Developed an ADAS Lane Assist module using Unity 3D and Python OpenCV. The project simulates real-time lane detection and steering offset estimation through perspective transformation, HSV thresholding, and sliding window algorithms, forming a base for autonomous lane-keeping systems.

ADAS Lane Assist Simulation Using Unity 3D & OpenCV
📖 Overview

This project demonstrates the development of an Advanced Driver Assistance System (ADAS) lane assist module using a Unity 3D simulation environment integrated with a Python-OpenCV lane detection pipeline.
It detects lane markings, estimates the vehicle’s steering offset, and visualizes real-time lateral deviation — a foundational step toward autonomous lane-keeping.

🎯 Objective

To design and implement a system capable of:

Detecting road lanes from a simulated car’s front camera.

Calculating the vehicle’s lateral offset from the lane center.

Providing a continuous offset graph for steering correction or future AI training.

🧠 Methodology
1. Unity 3D Environment

Built a dynamic driving simulation with:

Functional car controller (steering, braking, acceleration).

Endless city environment with dynamic traffic.

Snapshot camera capturing sequential road images.

Snapshots saved as .png frames for image-processing input.

2. Python + OpenCV Lane Detection

Implemented in arvr_lanedetection.py, the script:

Loads sequential images using natsort.

Applies Perspective Transformation to obtain a bird’s-eye view.

Uses HSV Thresholding to isolate lane lines.

Implements a Sliding Window Algorithm to track lane curvature and boundaries.

Computes lateral offset between car center and lane center (in meters).

Displays live visual output and a Matplotlib offset graph in real time.

⚙️ Technologies Used
Component	Tools/Frameworks
Simulation	Unity 3D (C#)
Image Processing	Python, OpenCV
Visualization	Matplotlib
File Handling	NumPy, os, datetime, natsort
🧩 Features

✅ Real-time lane detection on simulated road images
✅ Perspective warp and HSV masking for robust detection
✅ Sliding window tracking for accurate lane curve fitting
✅ Live steering offset visualization and logging
✅ Graph export of offset over time (in meters)

📂 File Structure
.
├── Unity_Project/
│   └── Assets/
│       └── Snapshots/              # Generated road view images
├── arvr_lanedetection.py           # Python lane detection script
├── steering_offset_graphs/         # Auto-saved output graphs
├── README.md                       # Project documentation
└── group2_arvr_report.pdf          # Detailed project report

🚀 Usage

Run Simulation in Unity

Play the Unity scene to generate road snapshots in Assets/Snapshots/.

Execute Python Script

python arvr_lanedetection.py


Adjust HSV trackbars to fine-tune lane detection.

View the live lane overlay, mask windows, and real-time offset graph.

After execution, the script automatically saves the final offset graph in steering_offset_graphs/.

📊 Results

The system accurately tracked lane boundaries across multiple frames.

Offset values (in meters) showed consistent response to left/right deviations.

Output graph visualized frame-by-frame steering offset — suitable as input for a PID controller or AI training dataset.

🔮 Future Work

Integrate PID control in Unity to enable autonomous steering correction.

Train a Convolutional Neural Network (CNN) for end-to-end lane-keeping.

Test robustness under various lighting, weather, and road conditions.

👨‍💻 Authors

Ojas Anil Rathi – Simulation & Python Integration

Pranav P – Vehicle Dynamics and Data Handling

Ashutosh Somayaji – Image Processing & Visualization

Supervisor: Dr. Asha C. S, Department of Mechatronics, Manipal Institute of Technology
