# MoveMatch: Advanced Human Pose Analysis and Comparison 

## Introduction
MoveMatch is an advanced pose comparison technology designed to revolutionize the realm of human pose analysis. Leveraging Google's MediaPipe for pose estimation, MoveMatch offers a comprehensive solution for frame-by-frame pose comparison between a reference image and an input video. Its applications range from helping athletes refine their techniques to assisting yoga practitioners in achieving complex poses.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Frame Extraction**: Extracts frames and keypoints from both reference image and input video.
  
- **Key Point Mapping**: Utilizes Google's MediaPipe to create a detailed map marking 33 distinct keypoints with confidence scores.

- **Data Standardization**: Ensures the keypoints are standardized across various sources, reducing bias due to size or position variations.
  
- **Noise Reduction**: Applies a Savitzky-Golay (Savgol) filter for data smoothing.

- **Advanced Matching**: Uses weighted Euclidean distances for high-precision matching.

- **Versatility**: Applicable in diverse fields including sports, healthcare, and research.

---

## How It Works

1. **Extraction**: Frame and keypoints are extracted using MediaPipe.
  
2. **Processing and Standardization**: Coordinates are standardized to their respective bounding boxes.

3. **Further Normalization and Smoothing**: Coordinates are further normalized and smoothed using a Savgol filter.

4. **Scoring**: Calculations are made based on weighted Euclidean distances.

5. **Identifying the Match**: The frame that most closely matches the reference pose is identified.

For a deeper dive into the code, check out our [Technical Documentation](LINK_HERE).

---

## Installation

# Clone the repository
```git clone https://github.com/Xcompany129/MoveMatch.git```

# Navigate into the directory
```cd MoveMatch```

# Install dependencies
```pip install -r requirements.txt```

---

## Usage

# CLI
To run a sample:
```python movematch.py --video ./reference_videos/test_1.mp4 --reference reference_images/ref_1.png --output results/```

# Jupyter Notebook
Use the notebook for a detailed understanding [MoveMatch.ipynb](MoveMatch.ipynb)

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any additional queries or feedback, feel free to contact the maintainers. Thank you for considering MoveMatch for your pose analysis needs.


