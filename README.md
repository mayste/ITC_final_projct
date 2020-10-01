# ITC final project - Cyberbullying Detector

In today’s society, Cyberbullying is a very important social issue that needs to be taken into consideration. Over the last years, almost 60% of children aged 10-17 have reported being bullied or harassed over the Internet and via social media outlets, but 90% of these victims will never report the incidents. 
While we might think this phenomenon only affects individuals, business cyberbullying is also a serious problem. It can cause lost revenue, a decrease in employee morale and a downtick in a company’s persona and prestige.

In this project we tried to deal with this issue. We use Natural Language Processing (NLP) tools to analyze text data, more precisely tweets, and detect cyberbullying posts. This may help twitter or companies to detect cyberbullying posts and block their diffusion for instance.

## Project Structure:

![image (1)](https://user-images.githubusercontent.com/66407270/94777264-d911d100-03cb-11eb-9b01-222d1683b798.png)

In this GitHub you will find:

## Datasets :

- Hate speech : 9924 posts classified as hate speech or not.
- Labeled-tweet : 11091 tweets classified are offensive and non-offensive
- Public data (Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar): 24,783 tweets classified as hate_speech, offensive or neither, in this dataset we combined data under the labels hate_speech and offensive and added them to the Cyberbullying label in our final dataset.

We combine the 3 of them in order to get a balanced dataset.

## Notebooks:

  - Cleaning_data: In this notebook we cleaned the dataset in order to use it when we ran the model
  - Cleaning_data_bert: This notebook is specific for the BERT Deep Learning model we replace the url, numbers, users with the tokens: 'url', 'number', 'user'.
  - LSTM : We run a LSTM model on our dataset which achieve not a good result. The model overfit really quickly.
  - Machine Learning Models: You will find the basics machine learning models we tried (SVM, Naive Bayes, Random Forest Classifier)
  - BERT Model with Transformers: This notebook present the BERT Deep Learning model we choose for the final deployment of our application.

## Static/css & Templates

These files contains the designs we used for our application that you can run using the *flask_server.py* file. 
Note: You need the procfile to run the app. The fisrt run will be slow as it downloads the pretrained weigths of the BERT model.

## Requirements.txt :

In order to run our app you'll need to install all the following packages:
* absl-py==0.10.0
* astunparse==1.6.3
* cachetools==4.1.1
* certifi==2020.6.20
* chardet==3.0.4
* click==7.1.2
* dataclasses==0.7
* filelock==3.0.12
* Flask==1.1.2
* gast==0.3.3
* google-auth==1.21.2
* google-auth-oauthlib==0.4.1
* google-pasta==0.2.0
* googledrivedownloader==0.4
* grpcio==1.32.0
* gunicorn==20.0.4
* h5py==2.10.0
* idna==2.10
* importlib-metadata==2.0.0
* itsdangerous==1.1.0
* Jinja2==2.11.2
* joblib==0.16.0
* Keras-Preprocessing==1.1.2
* Markdown==3.2.2
* MarkupSafe==1.1.1
* numpy==1.18.5
* oauthlib==3.1.0
* opt-einsum==3.3.0
* packaging==20.4
* protobuf==3.13.0
* pyasn1==0.4.8
* pyasn1-modules==0.2.8
* pyparsing==2.4.7
* regex==2020.7.14
* requests==2.24.0
* requests-oauthlib==1.3.0
* rsa==4.6
* sacremoses==0.0.43
* scipy==1.4.1
* sentencepiece==0.1.91
* six==1.15.0
* tensorboard==2.3.0
* tensorboard-plugin-wit==1.7.0
* tensorflow==2.3.0
* tensorflow-estimator==2.3.0
* termcolor==1.1.0
* tokenizers==0.8.1rc2
* tqdm==4.49.0
* transformers==3.2.0
* urllib3==1.25.10
* Werkzeug==1.0.1
* wrapt==1.12.1
* zipp==3.2.0

## Installation:
```python
git clone https://github.com/mayste/ITC_final_projct.git
pip install -r requirments.txt
```
