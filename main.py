import pandas as pd
from model import NetworkTrafficModel

def main():
    # Sample data (replace with actual dataset)
    data = pd.DataFrame({
        'time_of_day': [1, 2, 3],
        'packet_count': [200, 450, 300],
        'bandwidth_usage': [10.5, 25.3, 18.8]
    })
    
    model = NetworkTrafficModel()
    model.train()
    predictions = model.predict(data)
    
    print("Predicted Traffic:")
    print(predictions)

if __name__ == "__main__":
    main()
