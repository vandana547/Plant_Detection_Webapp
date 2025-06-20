# Plant_Detection_Webapp
# 📸 Plant Type Detector

Detect plant types in your images using AI. Powered by Roboflow.

**Created by Team SnapFlora**  

👩‍💻 VANDANA  SHARMA (roll no: 2401010285)

👩‍💻 SAKSHI  GARG (roll no: 2401010178)

👩‍💻 JYOTI SURYAVANSHI (roll no:2401010291)

👩‍💻 POOJA  AGGARWAL(roll no: 2401010240)


## Features

* 🖼️ Upload images for plant type detection.
* 📹  Capture live video feed from your webcam for real-time plant type detection.
* 🔗 Seamless integration with Roboflow AI API.
* ⚙️ Settings for adjusting image quality and confidence threshold.
* ✅ Option to show bounding boxes around detected plants.
* 📁 Detection history to review previous analyses.
* 🚀 Downloadable result images with detections.

## Installation

Follow these steps to get the Plant Type Detector up and running on your local machine:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/vandana547/Plant_Detection_Webapp.git

    cd Plant-Type-Detector
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    * Note: Ensure you have a `requirements.txt` file listing all necessary packages, including `streamlit`, `Pillow`, `opencv-python`, and `inference-sdk`.

3.  **Add and configure your Roboflow API key:**
    * Obtain an API key from your Roboflow account.
    * In `streamlit_app.py`, the API key is expected as a Streamlit input. You can modify the `api_key = st.sidebar.text_input("Roboflow API Key", ...)` line if you prefer to use environment variables or Streamlit secrets.

## How to Run

1.  **Start the Streamlit application:**

    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Open in your browser:**

    Streamlit will provide a local URL (e.g., `http://localhost:8501`) that you can open in your web browser.

## Usage

The application has three main tabs:

* **Upload Image:** Upload an image from your computer for plant type detection.
* **Webcam Capture:** Capture an image from your webcam for real-time detection.
* **Detection History:** Review your previous detection results.

Use the sidebar to adjust settings like image quality and confidence threshold, and whether to display bounding boxes.

## Workflow

1.  **Input:**
    * Upload an image or capture from your webcam.
2.  **Detection:**
    * Click the "Detect Plants" button.
    * The input is sent to the Roboflow API for processing.
3.  **Results:**
    * Detected plant types and their confidence levels are displayed.
    * If enabled, bounding boxes are drawn around the detected plants.
    * You can download the result image.
4.  **History:**
    * View previous inputs and results in the "Detection History" tab.

## Supported Formats

* Images: `.jpg`, `.jpeg`, `.png`

## Acknowledgements

* [Streamlit](https://streamlit.io/): For providing the web framework.
* [Roboflow](https://roboflow.com/): For the plant detection API.
* [Pillow (PIL)](https://pillow.readthedocs.io/): For image processing.
* [OpenCV](https://opencv.org/): For webcam capture.
* [Inference SDK](https://docs.roboflow.com/inference/python): For interacting with the Roboflow API.

  ## 📽️Video Presentation Of Project
  


https://github.com/user-attachments/assets/e1b553a9-9502-497c-a317-dad933352ecb

## 🔗For Full Video Go to Drive Link
[Link](https://drive.google.com/file/d/1t-ZdrFfvVmU6vpNaT7tc-hJy4ZdXK3Ik/view?usp=drivesdk)

##  🔁WorkFlow

```mermaid
flowchart LR
    A[User Input] --> B{Input Method}
    B -->|Upload Image| C[Load Image File]
    B -->|Use Webcam| D[Capture Image from Webcam]
    C --> E[Save Temp Image]
    D --> E
    E --> F[Send to Roboflow API for Prediction]
    F --> G[Receive Predictions]
    G --> H[Draw Bounding Boxes on Image]
    H --> I[Display Results on Streamlit UI]
```
