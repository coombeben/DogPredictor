import json

import numpy as np
from PIL import Image
from onnxruntime import InferenceSession


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)


def topk(x: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """NumPy version of torch.topk"""
    assert k >= 1

    x = x.flatten()

    top_ind_unsorted = np.argpartition(x, -k)[-k:]
    top_x_unsorted = x[top_ind_unsorted]

    top_ind_sorted = np.argsort(top_x_unsorted)[::-1]
    top_x_sorted = top_x_unsorted[top_ind_sorted]

    return top_x_sorted, top_ind_unsorted[top_ind_sorted]


def transform_image(img: Image) -> np.ndarray:
    """Applies standard image transforms to PIL Image"""
    mean = np.array([0.4850, 0.4560, 0.4060])
    std = np.array([0.2290, 0.2240, 0.2250])

    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    img_np = np.array(img, dtype=np.float32) / 255
    img_np = (img_np - mean) / std

    return np.expand_dims(np.moveaxis(img_np, -1, 0), axis=0).astype(np.float32)


class Classifier:
    def __init__(self, model_path: str):
        self.ort_session = InferenceSession(model_path)

        with open('labels.json', 'r') as json_file:
            self.labels = json.load(json_file)

    def get_prediction(self, img: Image, k: int = 5) -> list[dict]:
        if k > 269:
            raise ValueError('K must be lower than the number of classes')
        img_transformed = transform_image(img)

        ort_inputs = {self.ort_session.get_inputs()[0].name: img_transformed}
        output = self.ort_session.run(None, ort_inputs)[0]

        top_probabilities, top_class_indices = topk(softmax(output), k=k)
        top_probabilities, top_class_indices = top_probabilities.flatten().tolist(), top_class_indices.flatten().tolist()

        return [
            {'class_name': self.labels[str(idx)], 'class_prob': prob}
            for idx, prob in zip(top_class_indices, top_probabilities)
        ]
