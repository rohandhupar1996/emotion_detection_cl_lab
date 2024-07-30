

# Emotion Detection Computational Linguistics Lab

This project implements an emotion detection model using various computational techniques. It includes preprocessing of data, feature extraction, model training, and evaluation.

## Table of Contents

1. [Files Description](#files-description)
2. [How to Run](#how-to-run)
3. [Dependencies](#dependencies)

## Files Description


# Model Evaluation Reports

## BiLSTM

**Final Results:**
- Accuracy: 0.3345
- F1-Score: 0.3270

### Classification Report
| Class   |   Precision |   Recall |   F1-Score |   Support |
|:--------|------------:|---------:|-----------:|----------:|
| anger   |        0.35 |     0.41 |       0.38 |       176 |
| disgust |        0.36 |     0.29 |       0.32 |       173 |
| fear    |        0.34 |     0.45 |       0.39 |       164 |
| guilt   |        0.29 |     0.26 |       0.27 |       155 |
| joy     |        0.3  |     0.45 |       0.36 |       162 |
| sadness |        0.34 |     0.33 |       0.33 |       151 |
| shame   |        0.3  |     0.1  |       0.15 |       164 |
### Confusion Matrix
![BiLSTM Confusion Matrix](outputs/confusion_matrix_BiLSTM.png)

## DistilBERT

**Final Results:**
- Accuracy: 0.6585
- F1-Score: 0.6609

### Classification Report
| Class   |   Precision |   Recall |   F1-Score |   Support |
|:--------|------------:|---------:|-----------:|----------:|
| anger   |        0.7  |     0.71 |       0.71 |       151 |
| disgust |        0.53 |     0.72 |       0.61 |       155 |
| fear    |        0.61 |     0.44 |       0.51 |       164 |
| guilt   |        0.7  |     0.63 |       0.66 |       173 |
| joy     |        0.56 |     0.64 |       0.6  |       176 |
| sadness |        0.79 |     0.78 |       0.78 |       164 |
| shame   |        0.85 |     0.75 |       0.8  |       162 |
### Confusion Matrix
![DistilBERT Confusion Matrix](outputs/confusion_matrix_DistilBERT.png)


## BERT-CNN

**Final Results:**
- Accuracy: 0.6751
- F1-Score: 0.6772

### Classification Report
| Class   |   Precision |   Recall |   F1-Score |   Support |
|:--------|------------:|---------:|-----------:|----------:|
| anger   |        0.74 |     0.67 |       0.7  |       151 |
| disgust |        0.56 |     0.55 |       0.56 |       155 |
| fear    |        0.45 |     0.73 |       0.56 |       164 |
| guilt   |        0.8  |     0.54 |       0.65 |       173 |
| joy     |        0.63 |     0.47 |       0.53 |       176 |
| sadness |        0.7  |     0.77 |       0.73 |       164 |
| shame   |        0.86 |     0.84 |       0.85 |       162 |
### Confusion Matrix
![BERT-CNN Confusion Matrix](outputs/confusion_matrix_BERT-CNN.png)


# Final Model Comparison (Sorted by Macro F1-Score)

| Model       | Macro F1-Score | Micro F1-Score   |
|:------------|:---------------|-----------------:|
| BiLSTM      | 0.327          |           0.327  |
| Naive Bayes | 0.5325         |           0.5322 |
| DistilBERT  | 0.6609         |           0.6609 |
| BERT-CNN    | 0.6772         |           0.6772 |




# Project Setup Instructions

## Setting Up the Virtual Environment

To ensure that the project dependencies are properly managed, follow these steps to set up a virtual environment.

### For Unix/MacOS

1. **Create a virtual environment:**
   ```sh
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```sh
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

### For Windows

1. **Create a virtual environment:**
   ```sh
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```sh
   .\venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

By following these instructions, you'll set up the virtual environment correctly, ensuring that all dependencies specified in `requirements.txt` are installed.

If you encounter any issues or have questions, please refer to the project documentation or contact the project maintainer.

### Main Files

- **main_nb.py**: This script runs the Naive Bayes algorithm. It handles the loading of data, preprocessing, training, and evaluation of the Naive Bayes model.
- **main_advance_models.py**: This script runs the advanced models (BiLSTM, DistilBERT, BERT-CNN). It includes functions for loading data, preprocessing, training, and evaluation of these models.

### Model Files

- **distilbert.py**: Contains the implementation and configuration of the DistilBERT model for text classification.
- **bilstm.py**: Contains the implementation and configuration of the BiLSTM model for text classification.
- **bert_cnn.py**: Contains the implementation and configuration of the BERT-CNN model for text classification.

### Preprocessing Files

- **advance_preprocessing.py**: Includes advanced preprocessing techniques used for preparing data before feeding it into the advanced models.
- **preprocess.py**: Handles basic preprocessing tasks required for data preparation such as tokenization, stopword removal, and stemming.

### Training and Evaluation Files

- **train_advance_models.py**: Script dedicated to training the advanced models (BiLSTM, DistilBERT, BERT-CNN).
- **evaluation.py**: Contains evaluation metrics and methods for assessing the performance of the models.
- **dummy_evaluation.py**: A simplified version of the evaluation script used for initial testing and validation.

### Others

- **modified_nb.py**: Contains a scratch numpy version of the Naive Bayes algorithm with custom adjustments for this project.
- **requirements.txt**: Lists all the dependencies required to run the scripts in this repository.

## How to Run

1. **Setup Environment**:
   - Ensure you have Python installed on your system.
   - Install the required dependencies using the following command:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run Naive Bayes**:
   - Execute the Naive Bayes script by running:
     ```bash
     python main_nb.py
     ```

3. **Run Advanced Models**:
   - To train and evaluate advanced models (BiLSTM, DistilBERT, BERT-CNN), run:
     ```bash
     python main_advance_models.py
     ```

## Dependencies

All necessary dependencies are listed in the `requirements.txt` file. Install them using:
```bash
pip install -r requirements.txt
