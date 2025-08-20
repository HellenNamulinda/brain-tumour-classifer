# Brain Tumour Classification Web App

This project provides a web-based tool for classifying brain tumours from MRI images using a deep learning model trained in TensorFlow.

## Features

- Upload an MRI scan image (JPG, PNG, JPEG)
- Predicts the type of brain tumour:
  - Glioma
  - Meningioma
  - Neurocitoma
  - Schwannoma
  - Outros
  - NORMAL (no tumour detected)
- Displays prediction and estimated model accuracy
- Simple, user-friendly interface built with Streamlit

## Getting Started

### Prerequisites

- Python 3.7+
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

### Usage

1. Clone the repository.
2. Ensure the trained model file is present at `models/brain_tumor_tensorflow_model.h5`.
3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```
4. Open the app in your browser, upload an MRI image, and view the prediction.

## Model Training

The model was trained using TensorFlow and Keras. Training code and data augmentation steps are provided in `brain-tumor-tensorflow-with-augmentation.ipynb`.

## File Structure

- `streamlit_app.py` - Main web app
- `models/brain_tumor_tensorflow_model.h5` - Trained model
- `test_data/` - Example MRI images for testing
- `requirements.txt` - Python dependencies

## License

See [LICENSE](LICENSE) for details.

## Acknowledgements

- MRI images and dataset sources
- TensorFlow, Streamlit, and other open-source libraries
