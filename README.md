

# Emotion Detection Computational Linguistics Lab

This project implements an emotion detection model using various computational techniques. It includes preprocessing of data, feature extraction, model training, and evaluation.

## Table of Contents

1. [Files Description](#files-description)
2. [How to Run](#how-to-run)
3. [Dependencies](#dependencies)

## Files Description



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
