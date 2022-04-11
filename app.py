from statistics import mode
from flask import Flask, render_template, request
import random
from model import predict_svm, predict_cnn

app = Flask(__name__, static_folder='static')


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/classify/<model>", methods=['POST'])
def classify(model=None):
    text = request.get_json()["text"]
    if model == 'svm':
        classname = predict_svm(text)
    else:
        classname = predict_cnn(text)

    return {"text": text, "class": classname, "model": model}


def main():
    app.run(debug=False)


if __name__ == "__main__":
    main()
