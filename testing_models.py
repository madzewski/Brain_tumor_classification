import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score, roc_auc_score, roc_curve
from keras.models import load_model
from utils.load_data import load_data

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_roc_curve(true_y, y_prob, roc_auc):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, color="darkorange",lw= 2, label="ROC curve (area = %0.3f)" % roc_auc,)
    plt.plot([0, 1], [0, 1], color="navy", lw= 2, linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

def create_pred_cnn(y):
    y_pred = [1 if i>=0.5 else 0 for i in y]
    return y,y_pred

def create_pred_ml(y):
    y_prob = [i[1] for i in y]
    y_pred = [1 if i>=0.5 else 0 for i in y_prob]
    return y_prob,y_pred


IMG_WIDTH = 240
IMG_HEIGHT = 240

test_path = './augmented_preprocessed_data/test'
X_test, y_test = load_data(test_path, preprocess = False, describe = True)

model_name ='models/ml_model_svm.pkl'

if model_name[-1] == '5':
    model = load_model(model_name)
    # model.evaluate(X_test, y_test)
    y_prob, y_pred = create_pred_cnn(model.predict(X_test))
else:
    model = joblib.load(model_name)
    y_prob, y_pred = create_pred_ml(model.predict_proba(X_test))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
print(f"Accuracy:{acc}  Precision:{prec}  Recall:{rec}  F1-score:{f1}  ROC:{roc}")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

plot_roc_curve(y_test, y_prob, roc)