from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardscalar_transform(X):
    return StandardScaler().fit_transform(X)

def minmaxscaler_transform(X):
    return MinMaxScaler().fit_transform(X)
