from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('nmodel.pkl')
vectorizer = joblib.load('nvectorizer.pkl')

# Initialize NLTK components
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans the input text."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'\w*\d\w*', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\n', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text

def preprocess_input(text):
    """Preprocess the input text for prediction."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)
    vectorized_text = vectorizer.transform([processed_text])
    return vectorized_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    
    # Preprocess and vectorize the input comment
    vectorized_text = preprocess_input(comment)
    prediction = model.predict_proba(vectorized_text)[0]
    
    result = {
        'toxic': prediction[0],
        'severe_toxic': prediction[1],
        'obscene': prediction[2],
        'threat': prediction[3],
        'insult': prediction[4],
        'identity_hate': prediction[5]
    }
    
    is_bullying = prediction[6]>=0.5

    return jsonify({'result': result, 'is_bullying': bool(is_bullying)})

if __name__ == '__main__':
    app.run()
