import pickle
import numpy as np
from flask import Flask, request, jsonify, make_response, render_template
from utils import seed_everything, jamo_sentence, build_matrix, preprocess_text
from inference import get_prediction



app = Flask(__name__)


@app.route('/', methods=['GET'])
def default_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    input_text = str(request.form.get('input_text'))
    try:
        if not input_text or len(input_text) > 300:
            return

        if request.method == 'POST':
            
            # Text Preprocessing
            preprocessed_text, embedding_matrix = preprocess_text(input_text)
            
            # Model Prediction
            answer = get_prediction(preprocessed_text, embedding_matrix)
            print("입력 받은 문자는: ", input_text)
            print("예측 정답은: " , answer)        

            return render_template('result.html', input_text=input_text, answer=answer)
    except:
        return render_template('index.html')
    
    return render_template('index.html')
    


if __name__ == '__main__':
    seed_everything(seed=42)
    app.run(host="0.0.0.0", port="8080", debug=False)