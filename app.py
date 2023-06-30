import json

from flask import Flask, request, render_template
from io import BytesIO
from PIL import Image
import timm
import torch


# TODO: Validate files


app = Flask(__name__)
model = timm.create_model(
    'convnextv2_tiny.fcmae_ft_in22k_in1k',
    checkpoint_path='/models/convnext_tiny.pth',
    num_classes=269
)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
with open('labels.json', 'r') as json_file:
    labels = json.load(json_file)


def get_prediction(img_bytes, k: int = 5) -> list[dict]:
    if k > 269:
        raise ValueError('K must be lower than the number of classes')

    img = Image.open(BytesIO(img_bytes))
    with torch.no_grad():
        output = model(transforms(img).unsqueeze(0))
    top_probabilities, top_class_indices = torch.topk(output.softmax(dim=1), k=k)
    top_probabilities, top_class_indices = top_probabilities.flatten().tolist(), top_class_indices.flatten().tolist()

    return [
        {'class_name': labels[str(idx)], 'class_prob': prob}
        for idx, prob in zip(top_class_indices, top_probabilities)
    ]


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        predictions = get_prediction(img_bytes)
        return render_template('predict.html', predications=predictions)


if __name__ == '__main__':
    app.run()
