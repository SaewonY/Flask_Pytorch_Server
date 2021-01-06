import torch
import numpy as np
from model import NeuralNet

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_prediction(text, embedding_matrix):

    model = NeuralNet(embedding_matrix) 
    model.to(torch.device("cpu"))
    model.load_state_dict(torch.load('./pytorch_model/best_model.pt'))
    model.eval()

    with torch.no_grad():
        pred = model(text)
        pred_logit = sigmoid(pred.detach().cpu().numpy())[0][0]
        pred_label = np.where(pred_logit>=0.5, 1, 0)

    answer = '긍정' if pred_label == 1 else '부정'
    return answer