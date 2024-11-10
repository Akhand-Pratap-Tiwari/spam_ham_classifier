from flask import Flask, request, jsonify
import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# Initialize Flask app
app = Flask(__name__)


def custom_preprocessor(text):
    review = re.sub('[^a-zA-Z]', ' ', text)  # Remove Non-Alphabet Characters
    review = review.lower().split()
    # review = review.split()
    ps = PorterStemmer()
    # Remove stopwords
    # Get the stems for each word
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)  # join everything again
    return review


# Load the saved vectorizer
count_vectorizer = joblib.load('count_vectorizer.pkl')
nb_model = joblib.load("nb_model.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the incoming JSON request
    data = request.get_json()
    text = data.get('text', '')

    # Transform the text using the loaded vectorizer
    transformed_text = count_vectorizer.transform([text])

    prediction = nb_model.predict(transformed_text)
    return jsonify({
        'prediction': str(prediction[0])
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Get the text from the incoming JSON request
    data = request.get_json()
    text_arr = data.get('text_arr', '')

    # Transform the text using the loaded vectorizer
    transformed_text = count_vectorizer.transform(text_arr)

    predictions = nb_model.predict(transformed_text)
    return jsonify({
        'predictions': list(predictions)
    })


if __name__ == '__main__':
    app.run(debug=True)
