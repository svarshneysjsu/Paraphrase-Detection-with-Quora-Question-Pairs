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