import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic operational risk data for banks.
    Features:
      - transaction_volume
      - staff_count
      - system_failures
      - fraud_cases
    Target:
      - risk_event: 1 if high risk, 0 otherwise
    """
    np.random.seed(42)
    transaction_volume = np.random.randint(1000, 10000, num_samples)
    staff_count = np.random.randint(10, 200, num_samples)
    system_failures = np.random.randint(0, 20, num_samples)
    fraud_cases = np.random.randint(0, 10, num_samples)
    
    risk_event = ((system_failures > 10) | (fraud_cases > 5)).astype(int)
    
    df = pd.DataFrame({
        "transaction_volume": transaction_volume,
        "staff_count": staff_count,
        "system_failures": system_failures,
        "fraud_cases": fraud_cases,
        "risk_event": risk_event
    })
    
    X = df.drop("risk_event", axis=1)
    y = df["risk_event"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Scale features for deep learning
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
