
# Speech Command Classification using CNN & MLP

This repository contains two different approaches for **speech command classification** using **Wav2Vec2.0 + MLP (Multi-Layer Perceptron)** and **Mel-Spectrogram + CNN (Convolutional Neural Network)**. Both models are trained on **Google's Speech Commands Dataset** to classify 10 spoken words.

##  Overview
- **Model 1: Wav2Vec2.0 + MLP**
  - Extracts features using **Wav2Vec2.0**.
  - Classifies using a **Multi-Layer Perceptron (MLP)**.
- **Model 2: Mel-Spectrogram + CNN**
  - Converts audio to **Mel Spectrograms**.
  - Classifies using a **Convolutional Neural Network (CNN)**.
- **Dataset**: **Google’s Speech Commands Dataset**.
- **Performance**: 
  - **Wav2Vec2.0 + MLP:** 94.46% accuracy.(too overfitting)
  - **Mel-Spectrogram + CNN:** 80% accuracy.

---

## Dataset
The dataset consists of `.wav` files for 10 speech commands:
- yes
- no
- up
- down
- left
- right
- on
- off
- stop
- go

 The dataset can be downloaded from: [Google's Speech Commands Dataset](https://www.tensorflow.org/datasets/community_catalog/huggingface/speech_commands).

---

##  Libraries Used
- `librosa` - Audio processing
- `transformers` - Hugging Face's Wav2Vec2.0 model
- `scikit-learn` - MLP classifier, train-test split, evaluation metrics
- `torch` - Deep learning computations (for CNN)
- `numpy` - Data handling
- `matplotlib` - Visualization of spectrograms

---
