import pickle
from text_preprocess import text_preprocess
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np

svm_model = pickle.load(open('model/svm.pkl', 'rb'))
cnn_model = keras.models.load_model('model/model_cnn')
tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))

def predict_svm(text):
    text = text_preprocess(text)
    ret = svm_model.predict([ text ])[0]
    return ret

def predict_cnn(text):
    class_label = ['CongNghe',
               'DoiSong',
               'DuLich',
               'GiaiTri',
               'GiaoDuc',
               'KhoaHoc',
               'KinhDoanh',
               'PhapLuat',
               'SucKhoe',
               'TheThao',
               'Xe',
              ]
    text = text_preprocess(text)
    x = tokenizer.texts_to_sequences([ text ])
    x = pad_sequences(x, maxlen=500)
    pred = cnn_model.predict(x)
    ret = class_label[np.argmax(pred)]
    return ret