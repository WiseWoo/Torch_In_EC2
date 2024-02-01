import boto3
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import json
from torchvision import transforms
import io
import base64

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.models as models



EC2_sw=True
Lambda_sw=False

from flask_cors import CORS



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # 추가적인 변환을 여기에 포함시킵니다.
])

# Define Model
store_num=4
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, store_num)

BUCKET_NAME = "extractsignboard"
# MODEL_NAME = os.environ.get('MODEL_NAME')
# MODEL_NAME = "0030_quantized_resnet18.pth"
MODEL_NAME = "0030_resnet18_trained_0.003.pth"
s3_client = boto3.client('s3' )

if not os.path.isfile('/tmp/'+MODEL_NAME):
    s3_client.download_file(BUCKET_NAME, MODEL_NAME, '/tmp/'+MODEL_NAME)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
if device == 'cpu':
    model.load_state_dict(torch.load("/tmp/"+MODEL_NAME, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load("/tmp/"+MODEL_NAME))
model.eval()

if EC2_sw:
    from flask import Flask, request, jsonify
    import base64
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route('/upload', methods=['POST','GET'])
    def upload():
        try:
            data = request.json
            image_data = data['image']
            image_data = image_data.split(",")[1]  # base64 인코딩된 부분만 추출
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # img = resize_image_to_tensor(img).unsqueeze(0)
            img = transform(img).unsqueeze(0)
            print(img.size())

            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            print("predicted", predicted)
            return jsonify({'predicted_class': predicted.item()}), 200
        except Exception as e:
            print(e)
            return jsonify({'error':str(e)}), 500

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)