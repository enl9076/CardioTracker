import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plot_helpers import *
from analysis_model_helpers import *

def get_data():
    df = pd.read_csv("data/BP_Data.csv")
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
    
    st.title("BP Data Analysis")
    '''#### This app is helpful for those with POTS or POTS-like symptoms'''

    # Create the sidebar
    st.sidebar.title("Options")    
    st.sidebar.write("Select Plots to view:")
    plot_opts= ["BP TImeline", "HR Timeline", "HR Density", "Pairplot"]
    for p in plot_opts:
        st.sidebar.checkbox(p, value=True, key=p.lower().replace(" ", "_"))
    st.sidebar.write("Select Analysis:")
    analysis = st.sidebar.radio("Analysis", ["SVM Model", "Logistic Regression"], index=1)
    
    # Main info display
    st.subheader("Data Summary")
    st.write(get_average_by_state(df))
    
    # Display the selected analysis results
    if analysis == "SVM Model":
        accuracy_score, y_test, y_pred, le = run_svm_model(df)
        st.subheader("SVM Model Results")
        st.write("Accuracy:", accuracy_score)
        report = generate_classification_report(y_test, y_pred, le)
        print(report)
    elif analysis == "Logistic Regression":
        y_pred, y_test, le, accuracy_score = run_logistic_regression_model(df)
        st.subheader("Logistic Regression Results")
        st.write("Accuracy:", accuracy_score)
        generate_classification_report(y_test, y_pred, le)
        
    # Only display plots if they are selected in the sidebar
    if st.session_state.get('hr_density', True):
        p = generate_density_plot(df)
        st.pyplot(p)
   
    if st.session_state.get('bp_timeline', True):
        for date, group in df_grouped:
            fig = generate_bp_plots(group)
            fig.update_layout(title=f'Blood Pressure on {date}')
            st.plotly_chart(fig) 
    
    if st.session_state.get('hr_timeline', True):
        for date, group in df_grouped:
            fig = generate_hr_plots(group)
            fig.update_layout(title=f'Heart Rate on {date}')
            st.plotly_chart(fig) 



if __name__ == "__main__":
    main()

