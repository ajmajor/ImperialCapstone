from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def linearRegression(X,Y):
    lr_model = LinearRegression()
    return lr_model.fit(X, Y)

def randomForest(X, Y, n_estimators=100):
    rf_model = RandomForestRegressor(n_estimators, random_state=42)
    return rf_model.fit(X, Y)

def gradientBoost(X, Y, n_estimators=100, learning_rate=0.1):
    gb_model = GradientBoostingRegressor(random_state=42)
    return gb_model.fit(X, Y)

def trainTestValidationSplit(X, Y, split):
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=split[0], random_state=42)
    X_test, X_val, Y_test, Y_val = train_test_split(X_temp, Y_temp, test_size=split[1], random_state=42)

    print("Size of training set: " + str(len(X_train)) + " samples")
    print("Size of test set: " + str(len(X_test)) + " samples")
    print("Size of validation set: " + str(len(X_val)) + " samples")

    X_out = {
        "train": X_train,
        "test": X_test,
        "val": X_val
    }
    
    Y_out = {
        "train": Y_train,
        "test": Y_test,
        "val": Y_val
    }

    return X_out, Y_out

def trainTestCycle(X_train, Y_train, X_test, Y_test):

    models = {
        "Linear": linearRegression,
        "Forest": randomForest,
        "Gradient": gradientBoost
    }

    outputs = {}
    predictions = {}
    scores = {}

    for model, func in models.items():
        print("Processing model: " + model)
        outputs[model] = func(X_train, Y_train)
        scores[model] = outputs[model].score(X_train, Y_train)
        predictions[model] = outputs[model].predict(X_test)
    
    return outputs, scores, predictions