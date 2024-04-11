from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, classification_report
from sklearn.inspection import permutation_importance

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import randint
import time

class MLModels():

    # The following class variables are default settings which may be overridden by instanciated class
    # Set a uniform random state for all processes
    randomState = 42
    # Default scaling of predictors using StandardScaler to True
    scalePredictors = True
    # Default 5-way split for K-fold cross-validation
    nSplits = 5
    # Inverse Regularisation range
    C = [1.0, 5.0, 10.0, 20.0]
    # max_depth range
    maxDepth = [None, 10, 20]
    # n_estimators
    nEstimators =  [50, 100, 200]
    # kernels
    kernels = ('linear', 'rbf')
    # learning_rate
    learningRate = [0.1, 0.5, 1.0, 2.0]

    def __init__(self, random_state=randomState, scale_predictors=scalePredictors, n_splits=nSplits, 
                 C=C, max_depth=maxDepth, n_estimators=nEstimators, kernels=kernels, learning_rate=learningRate):

        self.results = {
            "outputs": {},
            "train": {
                "predictions": {},
                "scores": {},
                "confusion": {}
            },
            "test": {
                "predictions": {},
                "scores": {},
                "confusion": {}
            }
        }

        # Override default class variables where applicable
        self.randomState = random_state
        self.scalePredictors = scale_predictors
        self.nSplits = n_splits
        self.C = C
        self.maxDepth = max_depth
        self.nEstimators = n_estimators
        self.kernels = kernels
        self.learningRate = learning_rate

    # In the following model definitions, regressor__ is used to ensure hyperparameters for tuning 
    # are passed to the regressor step of the model pipeline used for optimisation
    models = {
        # Logistic Regression Model
        "Logistic": {
            "model": LogisticRegression,
            "fixedParams": {
                "random_state": randomState,
                "class_weight": 'balanced',
                "penalty": 'l2'
            },
            "optParams": {
                "regressor__C": C
            },
            "featureImportance": False
        },
        # Random Forest Classifier
        "Forest": {
            "model": RandomForestClassifier,
            "fixedParams": {
                "random_state": randomState,
                "max_features": 'sqrt',
            },
            "optParams": {
                "regressor__n_estimators": nEstimators,
                "regressor__max_depth": maxDepth
            },
            "featureImportance": True
        },
        # Gradient Boosting Classifier
        "Gradient": {
            "model": GradientBoostingClassifier,
            "fixedParams": {
                "random_state": randomState,
                "max_features": 'sqrt',
            },
            "optParams": {
                "regressor__n_estimators": nEstimators,
                "regressor__max_depth": maxDepth,
                "regressor__learning_rate": learningRate
            },
            "featureImportance": True
        },
        # Support Vector Machine Classifier
        "SVC": {
            "model": SVC,
            "fixedParams": {
                "random_state": randomState,
                "class_weight": 'balanced',
            },
            "optParams": {
                "regressor__C": C,
                "regressor__kernel": kernels
            },
            "featureImportance": True
        }
    }

def trainTestCycle(X, y, columns):

    '''
    This function will evaluate the models specified in the MLModels class above on the data passed.
    A full grid search and a randomised grid search will both be conducted for comparison of attained
    hyperparameter optimisation
    '''
    myMLModels = MLModels()
    results = {}
    scores = {}

    for modelName,modelData in MLModels.models.items():
        print("Processing " + modelName + " model:")
        # print(modelData)

        # Create output dictionary for this model
        results[modelName] = {}
        scores[modelName] = {
            "featureImportances": {},
            "Grid": {
                "train": {
                    "featureImportances": {},
                    "scores": {},
                    "confusion": {}
                },
                "test": {
                    "featureImportances": {},
                    "scores": {},
                    "confusion": {}
                }
            }, 
            "Random": {
                "train": {
                    "featureImportances": {},
                    "scores": {},
                    "confusion": {}
                },
                "test": {
                    "featureImportances": {},
                    "scores": {},
                    "confusion": {}
                }
            }
        }

         # Create a pipeline with preprocessing (e.g., StandardScaler) and regressor
        pipeElements = [('regressor', modelData['model'](**modelData['fixedParams']))]
        if myMLModels.scalePredictors == True:
            pipeElements.insert(0, ('scaler', StandardScaler()))
        # print(pipeElements)
        pipeline = Pipeline(pipeElements)

        # Next, initialize cross-validation
        cv = KFold(n_splits=myMLModels.nSplits, shuffle=True, random_state=myMLModels.randomState)

        # We will need to optimise hyperparameters specified in modelData.optParams for the model
        # Fixed parameters are specified in modelData.fixedParams
        # We will try both GridSearchCV and RandomizedSearchCV for comparison
        for searchType, opt in {"Grid": True, "Random": False}.items():
            print(searchType + " Search...")
            startTime = time.time()
            results[modelName][searchType] = {}
             # Perform grid search with cross-validation 
            search = GridSearchCV(pipeline, modelData['optParams'], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)  if opt == True else RandomizedSearchCV(estimator=pipeline,
                                n_iter=5,
                                param_distributions=modelData['optParams'],
                                scoring = 'neg_mean_squared_error',
                                random_state=myMLModels.randomState)
            search.fit(X, y)

            # Get the best hyperparameters
            best_params = search.best_params_
            print("Best Hyperparameters:", best_params)

            results[modelName][searchType]['duration'] = time.time() - startTime
            results[modelName][searchType]['bestParams'] = best_params
            results[modelName][searchType]['bestEstimator'] = search.best_estimator_
            results[modelName][searchType]['CVResults'] = search.cv_results_
            results[modelName][searchType]['bestScore'] = search.best_score_
        
    # Perform cross-validation for each optimised pipeline
    for model, data in results.items():
        print("Model: " + model)
        # perform train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=myMLModels.randomState)

        for theModel, theEstimator in {"Grid": data['Grid']['bestEstimator'], "Random": data['Random']['bestEstimator']}.items():
            optimisedModel = theEstimator.fit(X_train, y_train)
            y_pred_train = optimisedModel.predict(X_train)
            y_pred_test = optimisedModel.predict(X_test)
            # Produce Confusion Matrices
            scores[model][theModel]["train"]["confusion"] = confusion_matrix(y_train, y_pred_train)
            scores[model][theModel]["test"]["confusion"] = confusion_matrix(y_test, y_pred_test)
            # Plot confusion matrices
            plots = {"Training data":scores[model][theModel]["train"]["confusion"], "Test data": scores[model][theModel]["test"]["confusion"]}
            for run, data in plots.items():
                ax = plt.subplot()
                sns.heatmap(data, annot=True, fmt='g', ax=ax)
                ax.set_xlabel('Predicted labels')
                ax.set_ylabel('True labels')
                ax.set_title('Confusion Matrix for ' + run)
                plt.show()
            # Compute scores
            scores[model][theModel]["train"]["scores"]["precision"] = precision_score(y_train, y_pred_train)
            scores[model][theModel]["test"]["scores"]["precision"] = precision_score(y_test, y_pred_test)
            scores[model][theModel]["train"]["scores"]["recall"] = recall_score(y_train, y_pred_train)
            scores[model][theModel]["test"]["scores"]["recall"] = recall_score(y_test, y_pred_test)
            scores[model][theModel]["train"]["scores"]["f1"] = f1_score(y_train, y_pred_train)
            scores[model][theModel]["test"]["scores"]["f1"] = f1_score(y_test, y_pred_test)
            if MLModels.models[model]['featureImportance'] == True:
                scores[model][theModel]["featureImportances"] = theEstimator.named_steps['regressor'].feature_importances_
    
    return results, scores

