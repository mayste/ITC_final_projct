import os
import numpy as np
from flask import Flask, render_template, request
from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer
from scipy.special import softmax

app = Flask(__name__)

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
weights = model.load_weights("model.h5")


@app.route('/')
def home():
    return render_template('/templates/home.html')


@app.route('/predict', methods=['POST'])
def predict_fun():
    message = request.form['message']
    tokenized = tokenizer([message])
    input_names = ['input_ids', 'token_type_ids', 'attention_mask']
    logits = model.predict({k: np.array(tokenized[k])[None] for k in input_names})[0]
    scores = softmax(logits, axis=1)[:, 1]
    if scores[0] > (1-scores[0]):
        return render_template('result.html', prediction='Cyberbulling')
    else:
        return render_template('result.html', prediction='non-Cyberbulling')


# new_tweet = "I hate black people, they are stupid and look like shit"
#
#
# @app.route('/predict_single', methods=['GET'])
# def predict_single():
#     """
#     This function predict a single value
#     :return: string
#     """
#
#     final_features = np.array([float(request.args.get('fixed acidity')), float(request.args.get('volatile acidity')),
#                                float(request.args.get('citric acid')), float(request.args.get('residual sugar')),
#                                float(request.args.get('chlorides')), float(request.args.get('free sulfur dioxide')),
#                                float(request.args.get('total sulfur dioxide')), float(request.args.get('density')),
#                                float(request.args.get('pH')), float(request.args.get('sulphates')),
#                                float(request.args.get('alcohol'))])
#
#     prediction = model.predict(final_features.reshape(1, -1))[0]
#     return f'The quality will be {int(round(prediction, 1))}'
#
#
# @app.route('/predict_multiple', methods=['GET'])
# def predict_multiple():
#     """
#     Multiple prediction API that receives json file in the body,
#     and returns a json file with predictions.
#     :return: json
#     """
#
#     # Validate the request body contains JSON
#     if request.is_json:
#         # Parse the JSON into a Python dictionary
#         req = request.json
#         final_features = pd.DataFrame(data=req['data'],
#                                       columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#                                                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#                                                'pH',
#                                                'sulphates', 'alcohol'])
#         # print(final_features)
#         prediction = model.predict(final_features)
#         response_body = []
#         for pred in prediction:
#             res = f'The quality will be {int(round(pred, 1))}'
#             response_body.append(res)
#         return make_response(jsonify(response_body))


if __name__ == "__main__":
    port = os.environ.get('PORT')
    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run(debug=True)
