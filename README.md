# Spam-Ham Classifier

This project is a machine learning-based spam/ham classifier designed to identify whether a given text message is spam or not. The classifier was developed by analyzing and comparing multiple machine learning models, ultimately selecting the Naive Bayes (NB) classifier due to its superior performance. The project includes a Flask application for serving the model and an interactive Jupyter notebook for testing the API endpoints.

## Project Features

- **Model Analysis and Comparison:** Evaluates multiple machine learning algorithms to determine the best-performing model.
- **Naive Bayes Classifier:** Implements a trained Naive Bayes model for spam classification.
- **Flask API:** Provides endpoints to classify single or batch inputs through HTTP requests.
- **API Testing Notebook:** Interactive testing of API endpoints using Python's `requests` library.
- **Preprocessing Artifacts:** Includes a pre-trained model and a count vectorizer for text processing. You can run and tweak the notebook for getting something different.

## Directory Structure
```
.
├── api_tester.ipynb              # Jupyter Notebook to test Flask API endpoints
├── count_vectorizer.pkl          # Pre-trained count vectorizer for text feature extraction
├── main.py                       # Flask application to serve the spam/ham classifier
├── nb_model.pkl                  # Trained Naive Bayes model for spam classification
├── Screenshot 2024-11-10 162751.png # Screenshot of report before adding additional preprocessor for language
├── spam.csv                      # Dataset containing spam and ham messages
└── spam_ham_classifier.ipynb     # Jupyter Notebook for model analysis, training, and evaluation
```

## File Descriptions

1. **`api_tester.ipynb`:**  
   A Jupyter Notebook to test the Flask API by sending requests and validating responses. It ensures the API functions as expected for single and batch predictions.

2. **`count_vectorizer.pkl`:**  
   The serialized count vectorizer object used for transforming text data into numerical features compatible with the trained model.

3. **`main.py`:**  
   The Flask application script hosting the spam/ham classifier. It exposes an HTTP API with endpoints for predicting whether input text is spam or not.

4. **`nb_model.pkl`:**  
   The serialized Naive Bayes classifier model trained on the spam/ham dataset.

5. **`spam.csv`:**  
   The dataset used to train and validate the spam/ham classifier. It contains labeled data with spam and ham messages.

6. **`spam_ham_classifier.ipynb`:**  
   A Jupyter Notebook for data exploration, preprocessing, model training, and evaluation. It also includes comparisons of different ML models.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - Flask
  - scikit-learn
  - pandas
  - requests (for testing API)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Akhand-Pratap-Tiwari/spam_ham_classifier.git
   cd spam-ham-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python main.py
   ```

4. Open the `api_tester.ipynb` notebook to test the API functionality.
   
## API Endpoints

- **`/predict`**: Accepts POST requests with a JSON body containing single text data for classification.
- **`/batch_predict`**: Accepts POST requests with a JSON body containing batch text data for classification.

## Results and Performance

- Achieved **98% accuracy** and an **F1-score of 0.93** on the test dataset.
- Selected Naive Bayes classifier as the best-performing model after analyzing multiple algorithms.

Feel free to contribute by opening issues or pull requests to enhance the project! 🎉
