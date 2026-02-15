from data_loader import generate_synthetic_data, scale_data
from model_ml import train_ml_model, evaluate_model
from model_dl import train_dl_model
from utils import greet

def main():
    greet("Operational Risk ML + DL Project")
    
    # Generate synthetic data
    X_train, X_test, y_train, y_test = generate_synthetic_data(num_samples=1000)
    
    # ----- Machine Learning -----
    ml_model = train_ml_model(X_train, y_train)
    evaluate_model(ml_model, X_test, y_test)
    
    # ----- Deep Learning -----
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    dl_model = train_dl_model(X_train_scaled, y_train, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
