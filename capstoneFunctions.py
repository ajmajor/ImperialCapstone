# scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# PyTorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from skorch import NeuralNetClassifier
from sklearn.neural_network import MLPRegressor

# general imports
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import randint
import time
from itertools import product

# Define the FNN model class
class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learningRate=0.001, nEpochs=10, threshold=0.5):
        super(FNN, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.learningRate = learningRate
        self.nEpochs = nEpochs
        self.threshold = threshold

    def forward(self, x):
        hidden_output = self.sigmoid(self.hidden_layer(x))
        final_output = self.sigmoid(self.output_layer(hidden_output.float()))
        return final_output

class MLModels():
    
    # The following class variables are default settings which may be overridden by instanciated class
    # Set a uniform random state for all processes
    randomState = 42
    # Default to K-fold cross-validation, with the option for Stratified K-fold
    stratifiedKF = False
    # Default 5-way split for K-fold cross-validation
    nSplits = 5
    # Inverse Regularisation tuning options
    C = [1.0, 5.0, 10.0, 20.0]
    # max_depth range tuning options
    maxDepth = [None, 10, 20]
    # n_estimators tuning options
    nEstimators =  [50, 100, 150, 200]
    # kernels tuning options
    kernels = ('rbf', 'poly')
    # degree for polynomial kernel
    degree = [2,3,4]
    # learning_rate tuning options
    learningRate = [0.1, 0.5, 1.0, 1.5]

    def __init__(self, random_state=randomState, n_splits=nSplits, 
                 C=C, max_depth=maxDepth, n_estimators=nEstimators, 
                 kernels=kernels, degree = degree, learning_rate=learningRate,
                 stratified_kf = stratifiedKF):

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
        self.randomState = random_state#
        self.nSplits = n_splits
        self.C = C
        self.maxDepth = max_depth
        self.nEstimators = n_estimators
        self.kernels = kernels
        self.degree = degree
        self.learningRate = learning_rate
        self.stratifiedKF = stratified_kf

    # In the following model definitions, regressor__ is used to ensure hyperparameters for tuning 
    # are passed only to the regressor step of the model pipeline used for optimisation
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
                # We effectively don't want to optimise the logistic regression, so only pass the default C value
                # We do this as we still want to take advantage of the automated fitting of the model
                "regressor__C": [1.0]
            },
            "featureImportance": False,
            "optimiseHyperparameters": True,
            "scalePredictors": False
        },
        # K-Nearest Neighbours
        "KNN": {
            "model": KNeighborsClassifier,
            "fixedParams": {},
            "optParams": {
                "regressor__weights": ['uniform', 'distance'],
                "regressor__n_neighbors": [3,4,5,8]
            },
            "featureImportance": False,
            "optimiseHyperparameters": True,
            "scalePredictors": True
        },
        # Random Forest Classifier
        "Forest": {
            "model": RandomForestClassifier,
            "fixedParams": {
                "random_state": randomState,
                "max_features": 'sqrt',
            },
            "optParams": {
                # We effectively don't want to optimise the random forest classifier, so only pass the default values
                # We do this as we still want to take advantage of the automated fitting of the model
                "regressor__n_estimators": [100],
                "regressor__max_depth": [None]
            },
            "featureImportance": True,
            "optimiseHyperparameters": True,
            "scalePredictors": False
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
            "featureImportance": True,
            "optimiseHyperparameters": True,
            "scalePredictors": False
        },
        # Support Vector Machine Classifier
        "SVC": {
            "model": SVC,
            "fixedParams": {
                "random_state": randomState,
                "class_weight": 'balanced',
            },
            "optParams": {
                # We effectively don't want to optimise the SVC classifier, so only pass the default values
                # We do this as we still want to take advantage of the automated fitting of the model
                "regressor__C": [1.0],
                "regressor__kernel": kernels,
                "regressor__degree": degree
            },
            "featureImportance": False,
            "optimiseHyperparameters": True,
            "scalePredictors": False
        }
    }

    def holdOutSplit(X,y, split=0.2, stratify=False, randomState=randomState):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=stratify, random_state=randomState)
        return X_train, y_train, X_test, y_test
    
    def trainFNN(X,y, nEpochs=10, learningRate=0.001, threshold=0.5):

        # We need to convert our training data to tensors for PyTorch
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        train_dataset = TensorDataset(X_tensor, y_tensor)

        # Set hyperparameters
        input_dim = len(X.columns)  # Input size is number of feaatures in training set
        hidden_dim = 121    # Size of the hidden layer
        output_dim = 1       # Binary classification

        # Instantiate the FNN model
        model = FNN(input_dim, hidden_dim, output_dim, nEpochs=nEpochs, learningRate=learningRate, threshold=threshold)

        # Define loss function and optimizer
        criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
        optimizer = torch.optim.SGD(model.parameters(), lr=model.learningRate)  # Stochastic Gradient Descent

        batch_size = 32
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        n_epochs = model.nEpochs
        for epoch in range(1, n_epochs + 1):
            for data, target in train_loader:
                # Forward pass
                optimizer.zero_grad()
                output = model(data.view(-1, input_dim))
                loss = criterion(output, target.float().unsqueeze(1))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch}/{n_epochs}] - Loss: {loss.item():.4f}")

        # Save the trained model
        # torch.save(model.state_dict(), 'fnn_model.pth')
        print("Model saved successfully!")
        return model

def trainTestCycle(X, y, stratifiedKF = False, threshold=0.5):

    '''
    This function will evaluate the models specified in the MLModels class above on the data passed.
    A hold-out sample will be partitioned for validation, and the remaining data used for K-Fold cross validation
    A full grid search and a randomised grid search will both be conducted for comparison of attained
    hyperparameter optimisation

    Additionally, a PyTorch FNNmodel will be trained and validated for comparison to the other supervised ML models
    '''
    myMLModels = MLModels(stratified_kf=stratifiedKF)
    results = {}
    # featureNames = X.columns

    # Perform initial hold-out split, 0.2 default split will suffice
    X_train, y_train, X_val, y_val = MLModels.holdOutSplit(X,y, stratify=y, randomState=MLModels.randomState) if stratifiedKF == True else MLModels.holdOutSplit(X,y, randomState=MLModels.randomState)
    print("Number of training positives: " + str(np.count_nonzero(y_train)))
    print("Number of validation positives: " + str(np.count_nonzero(y_val)))
    # Next, initialize cross-validation K-fold or Stratified K-fold if the dataset is imbalanced
    # We will create an outer and inner cross validation set for the hyperparameter optimisation and scoring to limt overfitting
    outer_split = myMLModels.nSplits - 2 if myMLModels.nSplits > 3 else myMLModels.nSplits
    cv_outer = KFold(n_splits=outer_split, shuffle=True, random_state=MLModels.randomState) if myMLModels.stratifiedKF== False else StratifiedKFold(n_splits=outer_split, shuffle=True, random_state=MLModels.randomState)
    cv_inner = KFold(n_splits=myMLModels.nSplits, shuffle=True, random_state=MLModels.randomState) if myMLModels.stratifiedKF== False else StratifiedKFold(n_splits=myMLModels.nSplits, shuffle=True, random_state=MLModels.randomState)

    for modelName,modelData in MLModels.models.items():
        print("Processing " + modelName + " model:")

        # Create output dictionary for this model
        results[modelName] = {}

         # Create a pipeline with preprocessing (e.g., StandardScaler) and regressor
        pipeElements = [('regressor', modelData['model'](**modelData['fixedParams']))]
        if modelData['scalePredictors'] == True:
            pipeElements.insert(0, ('scaler', StandardScaler()))
        pipeline = Pipeline(pipeElements)
        
        # If we are not optimising hyperparameters, call cross_correlation method on the pipeline
        if modelData['optimiseHyperparameters'] == False:
            #search = cross_validate(pipeline, X,y, cv=cv_inner, scoring='neg_mean_squared_error', return_estimator=True)
            startTime = time.time()
            results[modelName]['cross_val_score'] = cross_val_score(pipeline, X_train, y_train, scoring='f1', cv=cv_outer, n_jobs=-1)
            duration = time.time() - startTime
            results[modelName]['duration'] = duration
            # fit the un-optimised model?
        else:
            # We will need to optimise hyperparameters specified in modelData.optParams for the model
            # Fixed parameters are specified in modelData.fixedParams
            # We will try both GridSearchCV and RandomizedSearchCV for comparison
            # del scores[modelName]['scoreData']
            for searchType, opt in {"Grid": True, "Random": False}.items():
                print(searchType + " Search...")
                results[modelName][searchType] = {}
                startTime = time.time()
                # Perform grid search with cross-validation 
                search = GridSearchCV(pipeline, modelData['optParams'], cv=cv_inner, scoring='f1', n_jobs=-1)  if opt == True else RandomizedSearchCV(estimator=pipeline,
                                    n_iter=5, cv=cv_inner,
                                    param_distributions=modelData['optParams'],
                                    scoring = 'f1',
                                    random_state=MLModels.randomState)
                #print("Fitting optimised " + searchType + " search model...")
                search.fit(X_train, y_train)
                print("Number of training positives: " + str(np.count_nonzero(y_train)))
                print("Number of validation positives: " + str(np.count_nonzero(y_val)))
                # Get the best hyperparameters
                duration = time.time() - startTime
                results[modelName][searchType] = getBestHyperparameters(search)
                results[modelName][searchType]['duration'] = duration
                results[modelName][searchType]['cross_val_score'] = cross_val_score(search, X_train, y_train, scoring='f1', cv=cv_outer, n_jobs=-1)
                y_pred = search.predict(X_val)
                # Produce Confusion Matrix
                results[modelName][searchType]['validation'] = {}
                results[modelName][searchType]['validation']["confusion"] = confusionMatrix(y_val, y_pred, " ".join([modelName, searchType]))
                # ... and scoring metrics
                results[modelName][searchType]['validation']["precision"] = precision_score(y_val, y_pred)
                results[modelName][searchType]['validation']["recall"] = recall_score(y_val, y_pred)
                results[modelName][searchType]['validation']["f1"] = f1_score(y_val, y_pred)

    # Now try the FNN
    startTime = time.time()
    results['FNN'] = {}

    theModel = MLModels.trainFNN(X_train, y_train, nEpochs=10, learningRate=0.001, threshold=threshold)
    duration = time.time() - startTime
    results['FNN']['duration'] = duration
    
    # Create a DataLoader for validation data# We need to convert our training data to tensors for PyTorch
    X_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    val_dataset = TensorDataset(X_tensor, y_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)

    # Put the model in evaluation mode
    theModel.eval()

    # Initialize lists to store predictions and ground truth labels
    all_predictions = []
    all_targets = []
    all_outputs = []

    # Iterate through the validation DataLoader
    with torch.no_grad():
        for data, target in val_loader:
            output = theModel(data.view(-1, len(X.columns)))
            predictions = (output >= threshold).float()  # Convert probabilities to binary predictions
            all_predictions.extend(predictions.tolist())
            all_targets.extend(target.tolist())
            all_outputs.extend(output.tolist())
    # Calculate F1 score
    f1 = f1_score(all_targets, all_predictions, average='binary')  # Set average='binary' for binary classification
    print(f"F1 Score on Validation Data: {f1:.4f}")
    # Calculate roc_auc
    roc_auc = roc_auc_score(all_targets, all_predictions)
    print(f"ROC_AUC Score on Validation Data: {roc_auc:.4f}")
    # Calculate precision & recall
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)

    results['FNN']['scores'] = {}
    results['FNN']['scores']['f1'] = f1
    results['FNN']['scores']['roc_auc'] = roc_auc
    results['FNN']['scores']['precision'] = precision
    results['FNN']['scores']['recall'] = recall
    results['FNN']['confusion'] = confusionMatrix(all_targets, all_predictions, "FNN")
    results['FNN']['estimator'] = theModel

    return results, X_val, y_val

def getBestHyperparameters(search):
    '''
    Extract selected output attributes of GridSearchCV and RandomizedSearchCV functions for analysis
    '''
    result = {}
    best_params = search.best_params_
    print("Best Hyperparameters:", best_params)
    result['bestParams'] = best_params
    result['bestEstimator'] = search.best_estimator_
    result['CVResults'] = search.cv_results_
    result['bestScore'] = search.best_score_
    return result

def correlationMatrix(predictors):
    '''
    Plot a correlation matrix for the supplied set of features
    '''
    cm = plt.figure(figsize=(19, 15))
    plt.matshow(predictors.corr(), fignum=cm.number)
    plt.xticks(range(predictors.select_dtypes(['number']).shape[1]), predictors.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(predictors.select_dtypes(['number']).shape[1]), predictors.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
    return

def confusionMatrix(targets, predictions, name, show=True):
    ''' 
    Calculate the confusion matrx for the provided targets and predictions; optionally print it
    Default is to print
    '''

    print("Number of actual positives: " + str(np.count_nonzero(targets)))
    print("Number of predicted positives: " + str(np.count_nonzero(predictions)))

    cm = confusion_matrix(targets, predictions)

    if show == True:
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix for ' + name)
        plt.show()

    return cm

def trainFNNOnly(X,y, learningRate=0.001, nEpochs=10, threshold=0.5, display=False):
    # Test function to train only the PyTorch FNN
    myMLModels = MLModels(stratified_kf=True)
    X_train, y_train, X_val, y_val = MLModels.holdOutSplit(X,y, stratify=y, randomState=MLModels.randomState)

    startTime = time.time()
    model = NeuralNetClassifier(module=FNN)

    # Define the range for hidden layer sizes
    hidden_layer_sizes = [11,33,55,121]  # Adjust the step size as needed
    # Generate all combinations of hidden layer sizes
    all_hidden_sizes = list(product(hidden_layer_sizes))
    param_grid = {
        'learning_rate': ['adaptive', 'constant'],  # Learning rates
        'batch_size': [32, 64, 128],  # Batch sizes
        #'threshold': [0.2,0.25,0.3,0.35,0.4,0.45,0.5],
        'hidden_layer_sizes': all_hidden_sizes,
        'activation': ['logistic', 'relu'],
        'solver': ['lbfgs', 'sgd'],
        'alpha': [0.0001, 0.0005]
    }
#    # Create the GridSearchCV object
    grid_search = GridSearchCV(
        estimator=MLPRegressor(max_iter=100, n_iter_no_change=10),
        param_grid=param_grid,
        cv=3,
        verbose=5,
        n_jobs=-1
    )
#    # Fit the grid search to your data
    grid_search.fit(X_train, y_train.ravel())
    # Create the GridSearchCV object
    #grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
    #grid_search.fit(X_train, y_train)
    print(grid_search)
    results = {}
    # Get the best hyperparameters
    duration = time.time() - startTime
    results = getBestHyperparameters(grid_search)
    results['duration'] = duration
    results['cross_val_score'] = cross_val_score(grid_search, X_train, y_train, scoring='f1', cv=3, n_jobs=-1)
    predictions = grid_search.predict(X_val)
    y_pred = (predictions >= threshold)
    # Produce Confusion Matrix
    results["confusion"] = confusionMatrix(y_val, y_pred, "FNN")
    #return results
    print("Starting FNN training...")
    theModel = MLModels.trainFNN(X_train, y_train, learningRate=learningRate, nEpochs=nEpochs, threshold=threshold)
    duration = time.time() - startTime
    print("Duration of training: " + str(duration))

    # Create a DataLoader for validation data# We need to convert our training data to tensors for PyTorch
    X_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    val_dataset = TensorDataset(X_tensor, y_tensor)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)

    # Put the model in evaluation mode
    theModel.eval()

    # Initialize lists to store predictions and ground truth labels
    all_predictions = []
    all_targets = []
    all_outputs = []

    # Iterate through the validation DataLoader
    with torch.no_grad():
        for data, target in val_loader:
            output = theModel(data.view(-1, len(X.columns)))
            predictions = (output >= threshold).float()  # Convert probabilities to binary predictions
            all_predictions.extend(predictions.tolist())
            all_targets.extend(target.tolist())
            all_outputs.extend(output.tolist())
    if display == True:
        print(all_predictions)
        print(all_targets)
        print(all_outputs)
    # Calculate F1 score
    f1 = f1_score(all_targets, all_predictions, average='binary')  # Set average='binary' for binary classification
    print(f"F1 Score on Validation Data: {f1:.4f}")
    # Calculate roc_auc
    roc_auc = roc_auc_score(all_targets, all_predictions)
    print(f"ROC_AUC Score on Validation Data: {roc_auc:.4f}")
    confusionMatrix(all_targets, all_predictions, 'FNN')