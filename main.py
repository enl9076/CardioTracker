import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plot_helpers import *
from ml_model_helpers import *
from anomoly_model_helpers import *

def get_data(datafile = "data/BP_Data.csv"):
    '''Load and preprocess the blood pressure and heart rate data from a CSV file.
    Args:
        datafile (str): Path to the CSV file containing the data.
    Returns: 
        pd.DataFrame: A DataFrame containing the preprocessed data with columns for date, systolic, diastolic, heart rate, and state (sitting/standing).
    '''
    df = pd.read_csv(datafile)
    df['DateTime'] = df['Date'] + ' ' + df['Time'] 
    df['DateTime'] = pd.to_datetime(df['DateTime'], dayfirst=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    # Rename columns
    df.rename(columns={
        'SYS(mmHg)': 'SYS',
        'DIA(mmHg)': 'DIA',
        'Pulse(Beats/Min)': 'HR',
        'Date': 'Date'
    }, inplace=True)
    df['state'] = df['Note'].apply(lambda x: 'Sitting' if 'Sitting' in str(x) else 'Standing' if 'Standing' in str(x) else 'Unknown')
    # Filter out the 'Unknown' state
    df = df[df['state'].isin(['Sitting', 'Standing'])]
    return df

def get_average_by_state(df):
    # Group by 'state' and calculate the averages
    averages = df.groupby('state').agg({
        'HR': ['mean', 'std', 'median', 'min', 'max'],
        'SYS': ['mean', 'std', 'median', 'min', 'max'],
        'DIA': ['mean', 'std', 'median', 'min', 'max']
    }).reset_index()
    return averages



def main():
    df = get_data()
    df_grouped = df.groupby(('Date'))
        
    st.title("🩺 Vital Signs Dashboard")
    st.write("This application helps you track and analyze your blood pressure and heart rate data, especially useful for monitoring conditions like POTS (Postural Orthostatic Tachycardia Syndrome).")

    # Create the sidebar
    st.sidebar.header("💓 CardioTracker", divider="gray")  
    uploaded_data = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Upload your own CSV file with blood pressure and heart rate data.")
    if uploaded_data is not None:
        df = get_data(uploaded_data)
    st.sidebar.write("Date Range")
    start_date = st.sidebar.date_input("Start Date", value=df['Date'].min(), min_value=df['Date'].min(), max_value=df['Date'].max())
    end_date = st.sidebar.date_input("End Date", value=df['Date'].max(), min_value=df['Date'].min(), max_value=df['Date'].max())
    #start_date = st.sidebar.selectbox("Start Date", options=df['Date'].unique(), index=len(df['Date'].unique()) - 1)
    #end_date = st.sidebar.selectbox("End Date", options=df['Date'].unique(), index=0)
    df = df[(df['Date'] >= str(start_date)) & (df['Date'] <= str(end_date))]    
    df_grouped = df.groupby(('Date'))  
    st.sidebar.write("Plot Selection")
    plot_opts= ["BP TImeline", "HR Timeline", "HR Density", "Pairplot"]
    for p in plot_opts:
        st.sidebar.checkbox(p, value=True, key=p.lower().replace(" ", "_"))
    st.sidebar.write("Select Analysis:")
    analysis = st.sidebar.radio("Analysis", ["SVM Model", "Logistic Regression", "Random Forest", "Anomoly Detection"], index=1)
    
    # Main info display
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg BP (SYS)", f"{df['SYS'].mean():.1f} mmHg", border=True)
    col2.metric("Avg BP (DIA)", f"{df['DIA'].mean():.1f} mmHg", border=True)
    col3.metric("Avg HR", f"{df['HR'].mean():.1f} bpm", border=True)
    st.subheader("Grouped Data Summary")
    st.write(get_average_by_state(df))
    
    bp_timeline_tab, hr_timeline_tab, analysis_tab = st.tabs(["BP Timeline", "HR Timeline", "Analysis"])

    # Display the selected analysis results
    with analysis_tab:
        st.subheader("Analysis Results")
        st.write("Selected Analysis:", analysis)
        if analysis == "SVM Model":
            accuracy_score, y_test, y_pred, le = run_svm_model(df)
            st.subheader("SVM Model Results")
            st.write("Accuracy:", accuracy_score)
            p=generate_confusion_matrix(y_test, y_pred, le)
            st.pyplot(p)
        elif analysis == "Logistic Regression":
            y_pred, y_test, le, accuracy_score = run_logistic_regression_model(df)
            st.subheader("Logistic Regression Results")
            st.write("Accuracy:", accuracy_score)
            generate_classification_report(y_test, y_pred, le)
        elif analysis == "Random Forest":
            y_pred, y_test, le, accuracy_score = run_random_forest_model(df)
            st.subheader("Random Forest Results")
            st.write("Accuracy:", accuracy_score)
            generate_classification_report(y_test, y_pred, le)
            p = generate_feature_importance_plot(df, y_pred, le)
            st.pyplot(p)
        elif analysis == "Anomoly Detection":
            st.subheader("Anomoly Detection Results")
            anomoly_df = run_anomaly_detection(df)
            st.write("Anomalies Detected:")
            st.write(anomoly_df)
            plots = plot_anomalies(anomoly_df)
            st.pyplot(plots)
        
        # Only display plots if they are selected in the sidebar
        #if st.session_state.get('hr_density', True):
        #    p = generate_density_plot(df)
        #    st.pyplot(p)
            
        if st.session_state.get('pairplot', True):
            p = generate_pairplot(df)
            st.pyplot(p)
    
    with bp_timeline_tab:
        st.subheader("Blood Pressure Timeline")
        st.write("This section shows the blood pressure readings over time.")
        if st.session_state.get('bp_timeline', True):
            for date, group in df_grouped:
                fig = generate_bp_plots(group)
                fig.update_layout(title=f'Blood Pressure on {date}')
                st.plotly_chart(fig) 
    
    with hr_timeline_tab:
        st.subheader("Heart Rate Timeline")
        st.write("This section shows the heart rate readings over time.")
        if st.session_state.get('hr_timeline', True):
            for date, group in df_grouped:
                fig = generate_hr_plots(group)
                fig.update_layout(title=f'Heart Rate on {date}')
                st.plotly_chart(fig) 



if __name__ == "__main__":
    main()

