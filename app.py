from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import Xception ,ResNet50
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

vocab = np.load('tokenizer.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}
print("+"*50)
print("vocabulary loaded")

model = load_model('model_14.h5')
Xception = load_model('Xception.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])

def after():

    global model, Xception, vocab, inv_vocab

    img = request.files['file1']

    img.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")
    embedding_size = 128
    vocab_size = len(vocab)
    max_len = 32

    
    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (299,299))
    image = np.expand_dims(image, axis=0)

    #image = np.reshape(image, (1,299,299,3))
    image = image/127.5
    image = image - 1.0
    
    incept = Xception.predict(image).reshape(1,2048)

    print("="*50)
    print("Predict Features")


    text_in = ['start']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'end':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)



    return render_template('after.html', data=final)

if __name__ == "__main__":
     app.run(debug=True)
