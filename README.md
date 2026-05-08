# Speech Impediment Detection

A machine learning tool for detecting dysarthria from speech recordings. The project uses audio preprocessing, feature extraction, and a TensorFlow convolutional neural network to classify whether a voice sample is likely to show signs of dysarthria. It is designed as a faster and more accessible screening aid for speech impairment detection.

> **Disclaimer:** This project is intended for educational and screening support purposes only. It is not a medical diagnosis tool and should not replace evaluation by a licensed speech-language pathologist or medical professional.

## Overview

This project analyzes voice samples and predicts whether the speaker is likely to have dysarthria. The workflow includes:

1. Loading and organizing dysarthria/non-dysarthria audio data
2. Cleaning audio recordings with noise reduction
3. Extracting frequency-based speech features from each recording
4. Training a TensorFlow CNN classifier
5. Evaluating the model with accuracy, precision, recall, F1 score, ROC/AUC, and confusion matrix
6. Running inference on new `.wav` and `.m4a` voice samples

## Features

- Real-time speech sample analysis pipeline
- Audio cleaning and normalization using `Librosa`, `SoundFile`, and `noisereduce`
- Feature extraction using MFCC-based frequency representations
- CNN-based binary classifier built with TensorFlow/Keras
- Evaluation using classification report, confusion matrix, ROC curve, and recall score
- Demo-ready inference on new `.wav` and `.m4a` recordings
- Svelte frontend support for a lightweight user-facing interface

## Tech Stack

### Machine Learning / Data Processing

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Librosa
- SciPy
- SoundFile
- noisereduce
- Matplotlib / Seaborn

### Frontend

- Svelte

### Dataset / Environment

- Kaggle Dysarthria Detection dataset
- Google Colab notebook workflow

## Model Pipeline

### 1. Dataset Setup

The notebook downloads the dysarthria dataset from Kaggle and organizes it into filtered audio folders:

```text
input-data/dysarthria-detection/
├── torgo_data/
├── filtered_torgo_data/
│   ├── dysarthria_female/
│   ├── dysarthria_male/
│   ├── non_dysarthria_female/
│   └── non_dysarthria_male/
└── data.csv
```

The dataset metadata is loaded with Pandas, and file paths are updated to point toward the cleaned audio directory.

### 2. Audio Cleaning

Each recording is loaded with `scipy.io.wavfile`, passed through `noisereduce`, and written back into the filtered dataset directory.

```python
rate, x = wavfile.read(record['filename'])
reduced_noise = nr.reduce_noise(y=x, sr=rate)
wavfile.write(new_filename, rate, reduced_noise)
```

This step helps reduce background noise before extracting speech features.

### 3. Feature Extraction

The project extracts 128 MFCC features from each voice recording using Librosa. The mean MFCC values are used as compact numerical representations of each audio sample.

```python
x, sr = librosa.load(record['filename'])
mean_mfcc = np.mean(
    librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128, htk=True),
    axis=1
)
```

Labels are converted into binary classes:

```text
0 = non-dysarthria
1 = dysarthria
```

The feature vectors are reshaped into a 2D CNN input format:

```python
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)
```

### 4. CNN Architecture

The classifier is a compact TensorFlow/Keras CNN:

```text
Input: 16 x 8 x 1
Conv2D: 32 filters, 3x3 kernel, ReLU
MaxPooling2D
Conv2D: 64 filters, 3x3 kernel, ReLU
MaxPooling2D
Flatten
Dense: 32 units, ReLU
Dense: 1 unit, Sigmoid
```

The model contains approximately 35K trainable parameters.

### 5. Training

The model is trained with:

- Optimizer: Adam
- Loss: Binary cross-entropy
- Metric: Accuracy
- Callback: ModelCheckpoint
- Callback: EarlyStopping with best-weight restoration

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

The trained model is saved for later inference:

```python
model.save('nishant-dysarthria-trained-model')
```

## Results

The model achieved strong validation and test performance:

```text
Accuracy:   0.97
Precision:  0.97
Recall:     0.97
F1-score:   0.97
```

The notebook also evaluates performance using:

- Classification report
- Confusion matrix
- ROC curve
- AUC score
- Recall score

## Running Inference

The notebook supports predictions on new audio samples.

### WAV Input

```python
x, sr = librosa.load(no_noise_wav_filename)
X_online = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=128), axis=1)
X_online = X_online.reshape(-1, 16, 8, 1)

prediction = nishant_trained_model.predict(X_online)
dysarthria_prob = prediction[0][0]
```

### M4A Input

`.m4a` files are converted to `.wav` before prediction:

```python
from pydub import AudioSegment

track = AudioSegment.from_file(m4a_file, format='m4a')
track.export(wav_filename, format='wav')
```

### Prediction Logic

```python
threshold = 0.5

if dysarthria_prob > threshold:
    print('The person is likely to have dysarthria.')
else:
    print('The person is likely to be healthy.')
```

## Installation

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn librosa scipy soundfile noisereduce scikit-learn tensorflow pydub ffmpeg
```

If running in Google Colab, upload your `kaggle.json` file before downloading the dataset.

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/speech-impediment-detection.git
cd speech-impediment-detection
```

### 2. Add Kaggle Credentials

Place `kaggle.json` in the expected Kaggle directory:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset

```bash
kaggle datasets download iamhungundji/dysarthria-detection
```

### 4. Run Notebook

Open the notebook in Google Colab or Jupyter and run the cells in order:

1. Dataset download and setup
2. Noise filtering
3. Feature extraction
4. Train/test split
5. CNN training
6. Evaluation
7. Saved-model inference

## Example Output

```text
prediction: 0.82
The person is likely to have dysarthria.
```

or

```text
prediction: 0.19
The person is likely to be healthy.
```

## Project Highlights

- Built a TensorFlow-based dysarthria detection model for speech impairment screening
- Processed 20,000+ voice recordings using Python audio-processing tools
- Cleaned and normalized recordings before extracting speech features
- Trained a CNN classifier on reshaped audio feature vectors
- Achieved 95%+ validation accuracy and approximately 97% test accuracy in notebook evaluation
- Added inference support for new voice samples, including `.wav` and `.m4a` formats
- Designed for integration with a lightweight Svelte frontend

## Future Improvements

- Add a backend API for uploading and classifying speech samples
- Connect the model to the Svelte frontend for real-time browser-based screening
- Add speaker-independent train/test splits to improve real-world generalization
- Expand evaluation across accents, microphones, and noisy environments
- Add model explainability tools to identify which speech features influence predictions
- Export the model to TensorFlow Lite for mobile or edge deployment

## License

This project is for educational and research purposes. Add a formal license file before public release.
