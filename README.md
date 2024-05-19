
# Emotion Detection Computational Linguistics Lab

This project implements an emotion detection model using various computational techniques. It includes preprocessing of data, feature extraction, model training, and evaluation.

## Project Structure

- **`code/`**: Contains all the Python scripts necessary for running the project.
  - `main_nb.py`: Main script that runs the model training and testing.
  - `preprocess.py`: Handles data preprocessing.
  - `evaluation.py`: For evaluating the model's performance.
  - `modified_nb.py`: Modified logic of the main notebook.
  - `dummy_evaluation.py`: A dummy evaluator for basic testing.
  - `post_process.py`: Post-processing of the model outputs.
- **`data/`**: Contains all datasets used in the project.
  - `ssec/`: Contains the training, validation, and test datasets in CSV format.
  - `isear/`: Includes datasets in both CSV and XLSX formats for training, validation, testing, and predictions.
- **`.git/`**: Git directory containing version control configurations and history.

## Prerequisites

- Python 3.x
- Pip (Python package installer)

## Installation

First, clone the repository to your local machine:

```bash
git clone <repository-url>
cd emotion_detection_cl_lab
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Navigate to the code directory and run the main script:

```bash
cd code
python main_nb.py
```

## Configuration

Ensure that the data paths in the scripts are set correctly to point to the respective directories under `data/`.

## Contributions

Feel free to fork the project, create a new branch, and submit pull requests.

## License

Specify the license under which the project is released.

## Contact

For more information, contact [your-email@domain.com].
