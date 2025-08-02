from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD, InterQuartileRangeAD, VolatilityShiftAD, SeasonalAD

def run_anomaly_detection(df):
    """
    Runs anomaly detection on the vital signs data using IsolationForest.
    Returns: The dataframe with an added 'anomaly' column (1=normal, -1=anomaly).
    """
    features = ['HR', 'SYS', 'DIA']
    X = df[features]

    # Fit IsolationForest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(X)

    # Optionally, return the anomaly scores as well
    df['anomaly_score'] = iso_forest.decision_function(X)
    return df

def plot_anomalies(df):
    """
    Plots HR vs. SYS, highlighting anomalies detected by IsolationForest.
    """
    normal = df[df['anomaly'] == 1]
    anomaly = df[df['anomaly'] == -1]

    fig, ax = plt.subplots()
    ax.scatter(normal['SYS'], normal['HR'], c='blue', label='Normal', alpha=0.6)
    ax.scatter(anomaly['SYS'], anomaly['HR'], c='red', label='Anomaly', alpha=0.8, marker='x')
    plt.xlabel('Systolic BP (SYS)')
    plt.ylabel('Heart Rate (HR)')
    plt.title('Anomaly Detection: HR vs. SYS')
    plt.legend()
    plt.tight_layout()
    return fig