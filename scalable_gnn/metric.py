from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

def accuracy(y_predict, y_true):

    y_predict = y_predict.to("cpu").detach().numpy()
    y_predict = np.argmax(y_predict, axis=1)
    y_ture = y_true.to("cpu").detach().numpy()
    accuracy = accuracy_score(y_ture, y_predict)

    return accuracy
def evaluate(logits, y):

    label_list, t_list = inference(logits, y)
    label_list = np.array(label_list)
    acc = Accuracy(t_list, label_list)
    recall = recall_score(t_list, label_list, average="samples")
    f1score = f1_score(t_list, label_list, average="samples")

    return acc, recall, f1score
def inference(logits, y):

    logits = logits.to("cpu")
    y = y.to("cpu")
    zs = logits.data.numpy()
    ts = y.data.numpy()
    labels = list(map(lambda x: (x >= 0.5).astype(int), zs))

    return labels, ts
def Accuracy(y_true, y_pred):
    
    count = 0

    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q

    return count / y_true.shape[0]
