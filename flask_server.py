import os
import numpy as np
from flask import Flask, render_template, request
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from scipy.special import softmax
from google_drive_downloader import GoogleDriveDownloader as gdd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_t():
    message = request.form['message']
    tokenized = tokenizer(message)
    input_names = ['input_ids', 'token_type_ids', 'attention_mask']
    logits = model.predict({k: np.array(tokenized[k])[None] for k in input_names})[0]
    scores = softmax(logits, axis=1)[:, 1]
    if scores[0] >= (1 - scores[0]):
        return render_template('result.html', prediction=1, variable=np.round(scores[0]*100, 3))
    else:
        return render_template('result.html', prediction=0, variable=np.round((1 - scores[0])*100, 3))


if __name__ == "__main__":
    gdd.download_file_from_google_drive(file_id='1sM9RNmt4kxvyX97SxB1Zv-WMkjSxBpqH',
                                        dest_path='./model.h5')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.load_weights('model.h5')
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run(debug=True)
