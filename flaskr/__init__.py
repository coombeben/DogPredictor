import os

from flask import Flask, request, render_template, redirect

from . import prediction


app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY')
)
model = prediction.Classifier('convnext_tiny.onnx')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()  # TODO: Validate files
        predictions = model.get_prediction(img_bytes)
        return render_template('predict.html', predications=predictions)
    elif request.method == 'GET':
        return redirect('/')


if __name__ == '__main__':
    app.run()
