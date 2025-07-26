from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

def run_svm_model(df):
    # Prepare data
    X = df[['HR', 'SYS', "DIA"]]  # Features: Heart Rate, Systolic, Diastolic
    y = df['state']  # Target: Sitting/Standing

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_test, y_pred, le

def run_logistic_regression_model(df):
    from sklearn.linear_model import LogisticRegression

    # Prepare data
    X = df[['HR', 'SYS', "DIA"]]  # Features: Heart Rate, Systolic, Diastolic
    y = df['state']  # Target: Sitting/Standing

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Logistic Regression model
    log_reg_model = LogisticRegression(random_state=42)
    log_reg_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = log_reg_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, y_test, le, accuracy

def run_random_forest_model(df):
    from sklearn.ensemble import RandomForestClassifier

    # Prepare data
    X = df[['HR', 'SYS', "DIA"]]  # Features: Heart Rate, Systolic, Diastolic
    y = df['state']  # Target: Sitting/Standing

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    
    return y_pred, y_test, le, accuracy

def generate_classification_report(y_test, y_pred, le):
    from sklearn.metrics import classification_report

    # Print classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    return report

def generate_confusion_matrix(y_test, y_pred, le):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
def generate_feature_importance_plot(df, y_pred, le):
    from sklearn.ensemble import RandomForestClassifier
    import seaborn as sns

    # Prepare data
    X = df[['HR', 'SYS', "DIA"]]  # Features: Heart Rate, Systolic, Diastolic
    y = df['state']  # Target: Sitting/Standing

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train Random Forest model to get feature importances
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y_encoded)

    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    return plt.gcf()  # Return the current figure object for display in Streamlit