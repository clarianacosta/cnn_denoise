# U-Net Image Denoising (Salt & Pepper)

A U-Net CNN (Keras/TensorFlow) project for "salt & pepper" image denoising. This model is trained to remove noise while preserving image details and includes a Gradio UI demo for real-time inference on full-size images.

This project was developed by **Clariana Costa** and **Lara Marques** for the "Advanced Topics in Computing" course.

![Full Size Denoising Results](results_comparison_fullsize.png)


## 🌟 Features

* **U-Net Architecture:** Employs a U-Net model, which uses skip-connections to preserve fine details and textures, resulting in high-fidelity image reconstruction.
* **Full-Size Inference:** The model is fully convolutional and built to accept inputs of `(None, None, 1)`. This allows it to process images of any resolution, not just the 128x128 size used during training.
* **Interactive Demo:** Run `app_gui.py` to launch a web-based Gradio interface where you can upload your own noisy images and see the denoised results in real-time.
* **Efficient Training Pipeline:** The `train_denoiser.py` script uses `tf.data` with data augmentation (random flips) to ensure the model generalizes well and avoids overfitting.

## 🚀 Setup & Usage

Follow these steps to set up the environment, download the data, train the model, and run the demo.

### 1. Clone the Repository

```bash
git clone https://github.com/clarianacosta/cnn_denoise.git
cd cnn_denoise
````

### 2. Install Dependencies

It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install from the requirements file
pip install -r requisitos.txt
```

*(If you don't have `requisitos.txt`, you can install manually: `pip install tensorflow opencv-python numpy gradio scikit-learn matplotlib kaggle`)*

### 4. Download & Organize the Dataset

This project uses the "Salt and Pepper Noise Images" dataset from Kaggle. You must have the Kaggle API installed and configured (`kaggle.json`).

```bash
# 1. Download the dataset files
# This will download 'salt-and-pepper-noise-images.zip'
kaggle datasets download -d rajneesh231/salt-and-pepper-noise-images

# 2. Unzip the file
unzip salt-and-pepper-noise-images.zip

# 3. Organize the folders
# The training script expects the data inside the `images/` directory.
# After unzipping, move and rename the data folders to match this structure:
images/
├── Noisy_folder/
│   ├── noisy_img_1.png
│   └── ...
└── Ground_truth/
    ├── clean_img_1.png
    └── ...
```
*(You will need to identify the noisy and clean folders from the unzipped Kaggle archive and rename them to `Noisy_folder` and `Ground_truth` respectively, placing them inside the `images/` directory.)*

## 🖥️ Running the Project

You must train the model first to generate the `.h5` file, as it is not included in the repository.

### Step 1: Train the Model

Run the training script. This will load the data from the `images/` folder, build the U-Net, and train it.

```bash
python train_denoiser.py
```

This script will:

1.  Load and preprocess the data.
2.  Build and compile the U-Net architecture.
3.  Train the model and save the best weights to **`denoiser_model.h5`**.
4.  Generate the output graphs: `training_history.png` and `results_comparison_fullsize.png`.

### Step 2: Run the Interactive Demo

Once `denoiser_model.h5` has been created, you can launch the Gradio web app.

```bash
python app_gui.py
```

This will start a local Gradio server. Open the URL (e.g., `http://127.0.0.1:7860`) in your browser to upload an image and see the model in action.

## 📈 Training Results

The model was trained with data augmentation, which intentionally makes the training task harder. This is why the validation metrics (orange) outperform the training metrics (blue), indicating **excellent generalization and no overfitting**.

## 📂 File Structure

```
.
├── app_gui.py                      # The Gradio web app (loads the model)
├── train_denoiser.py               # The main training script (generates the model)
├── requisitos.txt                  # Python dependencies
├── images/                         # Folder for training/test data
│   ├── Noisy_folder/
│   └── Ground_truth/
├── training_history.png            # Saved graph of training metrics (output)
├── results_comparison_fullsize.png # Saved example of denoising (output)
├── venv/                           # Virtual environment (ignored by git)
└── denoiser_model.h5               # The pre-trained model (ignored by git)
```
