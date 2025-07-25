from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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