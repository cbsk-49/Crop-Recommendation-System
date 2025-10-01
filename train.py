import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import pickle

def train_and_save_model():
    """
    Loads data, trains the model, and saves the model, scaler, and encoder.
    """
    # 1. Load and Prepare Data
    print("Loading data...")
    df = pd.read_csv("Crop_recommendation.csv")

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    df_processed = df.drop('label', axis=1)

    # 2. Scale Features
    print("Scaling features...")
    features = df_processed.drop('label_encoded', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_df['label_encoded'] = df_processed['label_encoded']

    # 3. Split Data
    X = scaled_df.drop('label_encoded', axis=1)
    y = scaled_df['label_encoded']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42)

    # 4. Define and Train Model
    print("Training the stacked model...")
    # Base models
    estimators = [
        ('lr', LogisticRegression(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='mlogloss')),  # Removed deprecated use_label_encoder
        ('rf', RandomForestClassifier(random_state=42))
    ]

    # Meta-model
    meta_model = LogisticRegression()

    # Stacking classifier
    stacked_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5)

    # Train the final model
    stacked_model.fit(X_train, y_train)
    print("Training complete.")

    # 5. Save the necessary objects
    with open('stacked_model.pkl', 'wb') as model_file:
        pickle.dump(stacked_model, model_file)
    print("Model saved to stacked_model.pkl")

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    print("Scaler saved to scaler.pkl")

    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)
    print("Label encoder saved to label_encoder.pkl")

if __name__ == '__main__':
    # Run the training process
    train_and_save_model()