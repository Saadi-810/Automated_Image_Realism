# AI Image Detector

## Overview
AI Image Detector is a Python application that uses a machine learning model to determine if an image is AI-generated or real. The application provides a user-friendly interface for uploading images and viewing classification results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-image-detector.git
   cd ai-image-detector
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. A GUI window will open. Click "Select Image" to upload an image file.
3. The application will display the original and processed images, along with the classification result and confidence score.

## File Structure
- `app.py`: Main application file containing the GUI and image classification logic.
- `requirements.txt`: List of required Python packages.
- `model/`: Directory containing the trained model and training code.
  - `final_model.h5`: The final trained model.
  - `model_train_code.py`: Script used to train the model.
- `Results & Analysis/`: Directory containing analysis results.
  - `confusion_matrix.png`: Confusion matrix of the model's performance.
  - `evaluation_results.txt`: Text file with evaluation metrics.
  - `model_architecture.txt`: Text file describing the model architecture.
  - `training_history.png`: Plot of the training history.

## Results and Analysis
The model's performance was evaluated using a confusion matrix, and the results are available in the `Results & Analysis/` directory. The evaluation metrics and training history are also provided for further analysis.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
