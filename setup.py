from setuptools import setup, find_packages

setup(
    name="motion_detector",
    version="0.1.0",
    description="YOLOv8-based real-time moving object detection (detection + tracking core)",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.2.0",
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
    ],
    python_requires=">=3.9",
)
