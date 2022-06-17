import os
import io
import torchvision.transforms as transforms
import json
import base64
from flask import Flask, render_template, request, redirect
from PIL import Image
from torchvision import models

with open("imagenet_classes.txt") as f:
    imagenet_class_index = [line.strip() for line in f.readlines()]

model = models.googlenet(pretrained=True)
model.eval()

app = Flask(__name__)


def classify(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model(tensor)
    except Exception:
        return -1, 'Error!'
    _, max_prob_index = outputs.max(1)

    return max_prob_index, imagenet_class_index[max_prob_index]


def transform_image(image_bytes):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return preprocess(image).unsqueeze(0)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')

        if not file:
            return

        img_bytes = file.read()

        image_string = base64.b64encode(img_bytes).decode()

        class_id, class_name = classify(image_bytes=img_bytes)

        return render_template('result.html', class_id=class_id, class_name=class_name, image_mime_type=file.content_type, image_base64=image_string)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
