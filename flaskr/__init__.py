import os
from io import BytesIO

from flask import Flask, request, render_template, redirect
from PIL import Image, UnidentifiedImageError

from . import prediction


app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY')
)
model = prediction.Classifier('pico_student.onnx')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['upload']
        img_bytes = file.read()  # TODO: Validate files

        try:
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        except UnidentifiedImageError:
            return render_template('error.html', error='Invalid image')

        predictions = model.get_prediction(img)
        return render_template('predict.html', predications=predictions)
    elif request.method == 'GET':
        return redirect('/')


if __name__ == '__main__':
    app.run()
