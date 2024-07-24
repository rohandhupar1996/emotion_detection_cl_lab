
from advance_preprocessing import load_and_preprocess_data

def main():
    train_path = 'data/train_data.csv'  # Update with your actual path if different
    test_path = 'data/test_data.csv'    # Update with your actual path if different

    X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_data(train_path, test_path)
    
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

if __name__ == "__main__":
    main()
