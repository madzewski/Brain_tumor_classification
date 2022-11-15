import numpy as np 
import joblib
import pprint

from sklearn import svm
from utils.load_data import load_data
from sklearn.model_selection import GridSearchCV

from utils.building_models import build_hog_svm_model
if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    
    param_grid = [
        {   'hogify__orientations': [10],
            'hogify__cells_per_block': [(3,3)],
            'hogify__pixels_per_cell': [(8, 8)],
            'classify': [
                svm.SVC(kernel='linear', C=1)
            ]
        }
    ]
        
    HOG_pipeline = build_hog_svm_model()


    train_path = './preprocessed_data/train'
    test_path = './preprocessed_data/test'

    X_train, y_train = load_data(train_path, start = 0, stop=2000, preprocess = False, disable_tqdm = True)

    X_test, y_test = load_data(test_path, start = 0, stop=200, preprocess = False, disable_tqdm = True)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    grid_search = GridSearchCV(HOG_pipeline, 
                           param_grid, 
                           cv=3,
                           n_jobs=1,
                           scoring='accuracy',
                           verbose=1,
                           return_train_score=True)
 
    grid_res = grid_search.fit(X_train, y_train)
    joblib.dump(grid_res, 'models/ml_model_hog_sgd_model_v2.pkl')

    print(grid_res.best_score_)
    pp.pprint(grid_res.best_params_)

    best_pred = grid_res.predict(X_test)
    print('Percentage correct: ', 100*np.sum(best_pred == y_test)/len(y_test))