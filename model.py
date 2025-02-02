import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'time_of_day': [1, 2, 3, 4, 5],
    'packet_count': [200, 450, 300, 500, 400],
    'bandwidth_usage': [10.5, 25.3, 18.8, 30.1, 22.7],
    'network_load': [0.5, 0.8, 0.6, 0.9, 0.7]
})

class NetworkTrafficModel:
    def __init__(self):
        self.model = RandomForestRegressor()
        self.scaler = StandardScaler()
    
    def train(self):
        X = data[['time_of_day', 'packet_count', 'bandwidth_usage']]
        y = data['network_load']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)
