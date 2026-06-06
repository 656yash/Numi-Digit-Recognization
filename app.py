import os
from flask import Flask, render_template, request, jsonify
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

MODEL_PATH = 'model/mnist_cnn.tflite'

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))

        # Preprocess
        image = image.convert('L').resize((28, 28))
        img_array = 255 - np.array(image)
        img_array = img_array.reshape(1, 28, 28, 1).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        return jsonify({
            'digit': int(np.argmax(pred)),
            'confidence': float(np.max(pred))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)