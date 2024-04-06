from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def linearRegression(X,Y):
    lr_model = LinearRegression()
    return lr_model.fit(X, Y)

def randomForest(X, Y, n_estimators=100):
    rf_model = RandomForestRegressor(n_estimators, random_state=42)
    return rf_model.fit(X, Y)

def gradientBoost(X, Y, n_estimators=100, learning_rate=0.1):
    gb_model = GradientBoostingRegressor(n_estimators, learning_rate, random_state=42)
    return gb_model.fit(X, Y)