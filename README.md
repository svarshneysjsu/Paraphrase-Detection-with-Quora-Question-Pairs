# Paraphrase Detection with Quora Question Pairs

In this project, an LSTM model (NLP technique) is used to determine whether two Quora questions are similar or not. The Fasttext and GloVe word embeddings are utilized to train the model.

## Folder Structure
- raw_data - This will contain all the raw dataset files collected from -
    - QQP Dataset - https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
- staging_data - This folder will be used as a staging area to store intermediate tables.
- clean_data - This will contain the final clean dataset along with the train test splits.

## Order of execution
- Data Cleaning
	- data_cleaning_integration
- Data Transformation
	- train_validation_split
	- data_transformation
	- merge_data
- Models
	- LSTM_using_Fasttext.ipynb
	- LSTM_using_Glove.ipynb
- Inference
	- inference.ipynb

# Paraphrase Detection with Quora Question Pairs

## Description

This project implements a machine learning model to detect paraphrases using the Quora Question Pairs dataset. The model determines whether two given questions have the same meaning, addressing the problem of duplicate content on platforms like Quora. The solution involves data preprocessing, feature engineering, and training advanced natural language processing (NLP) models, with a focus on Long Short-Term Memory (LSTM) architecture. Experiments with different models, including Gated Recurrent Units (GRU) and Recurrent Neural Networks (RNN), were conducted to select the best-performing model.

## Features

- **Data Preprocessing**: Tokenization, text cleaning, and stopword removal.
- **Feature Engineering**: Extraction of semantic and syntactic features from question pairs.
- **Model Training**: Implementation of LSTM, GRU, and RNN architectures for classification.
- **Model Selection**: Evaluation and selection of the best model based on performance metrics.
- **Evaluation Metrics**: Calculation of accuracy, F1 score, and precision-recall analysis.
- **Visualization**: Insights into data distribution and model performance.

## Usage

To use this project, follow these steps:

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook (optional, for exploring data and running experiments)

### Folder Structure

- raw_data - This will contain all the raw dataset files collected from -
    - QQP Dataset - https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
- staging_data - This folder will be used as a staging area to store intermediate tables.
- clean_data - This will contain the final clean dataset along with the train test splits.

### Steps
- **Data Cleaning**
	- data_cleaning_integration
- **Data Transformation**
	- train_validation_split
	- data_transformation
	- merge_data
- **Models**
	- LSTM_using_Fasttext.ipynb
	- LSTM_using_Glove.ipynb
- **Inference**
	- inference.ipynb

 ## Technologies Used

- **Programming Language**: Python
- **Machine Learning**: TensorFlow, Keras
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Text Processing**: NLTK, spaCy

## Contributing

This project was developed as part of the Data 270 course at San Jose State University. This is a team project for academic purposes. If you find any issues or have suggestions, feel free to open a GitHub issue.

## Contact

For questions or additional information, please reach out via:
- **GitHub Issues**: [Open an Issue](https://github.com/svarshneysjsu/Paraphrase-Detection-with-Quora-Question-Pairs/issues)

---

Hope this project serves as a valuable resource for understanding paraphrase detection using machine learning and NLP techniques.
