import os
import numpy as np
from flask import Flask, render_template, request
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from scipy.special import softmax

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_t():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.load_weights("model.h5")
    message = request.form['message']
    tokenized = tokenizer(message)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask']

    logits = model.predict({k: np.array(tokenized[k])[None] for k in input_names})[0]
    scores = softmax(logits, axis=1)[:, 1]
    if scores[0] >= (1-scores[0]):
        return render_template('result.html', prediction=1)
    else:
        return render_template('result.html', prediction=0)


# new_tweet = "I hate black people, they are stupid and look like shit"

if __name__ == "__main__":
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run(debug=True)
