#Set-ExecutionPolicy Unrestricted -Scope Process
import numpy as np 
import joblib
import pprint

from sklearn import svm
from utils.load_data import load_data
from utils.tools import save_model_params
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from utils.building_models import build_hog_svm_model, build_svm_model, build_pca_svm_model


def train_grid_search(model, param_grid, model_name ,X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=2, scoring='accuracy', 
                            verbose=4, return_train_score=True)
    grid_result = grid_search.fit(X_train, y_train)
    joblib.dump(grid_result, 'models/'+model_name)

    print(grid_result.best_score_)
    pp.pprint(grid_result.best_params_)
    

    return grid_result

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    
    param_grid_hog = [
        {   'hogify__orientations': [10],
            'hogify__cells_per_block': [(3,3)],
            'hogify__pixels_per_cell': [(8,8)],
            'classify': [
                svm.SVC(kernel='linear', C=1)
            ]
        }
    ]
    param_grid_pca = [
        {   'reduce_dim__n_components': [100,250,500,1000],
        # {   'reduce_dim__n_components': [0.95],
            'classify': [
                svm.SVC(kernel='linear', C=1)
            ]
        }
    ]
    param_grid_svm = [
        {'classify': [
                svm.SVC(kernel='linear', C=1),
                svm.SVC(kernel='rbf', C=1)
            ]
        }
    ]      
    model_hog = build_hog_svm_model()
    model_pca = build_pca_svm_model()
    model_svm = build_svm_model()

    train_path = './preprocessed_data/train'
    test_path = './preprocessed_data/test'

    X_train, y_train = load_data(train_path, start = 0, stop=2000, preprocess = False, disable_tqdm = True)
    X_test, y_test = load_data(test_path, start = 0, stop=150, preprocess = False, disable_tqdm = True)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    # X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    model_name = 'ml_model_hog_gs.pkl'
    adj_model = train_grid_search(model_hog, param_grid_hog, model_name, X_train, y_train)

    best_pred = adj_model.predict(X_test)
    print('Percentage correct: ', 100*np.sum(best_pred == y_test)/len(y_test))