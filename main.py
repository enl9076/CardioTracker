import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

def get_data():
    df = pd.read_csv("data/BP_Data.csv")
    # Rename columns
    df.rename(columns={
        'SYS(mmHg)': 'SYS',
        'DIA(mmHg)': 'DIA',
        'Pulse(Beats/Min)': 'HR',
        'Date': 'Date'
    }, inplace=True)
    return df

def generate_plot(df):
    fig = go.Figure()
    # Plot SYS, DIA, and PULSE in one plot
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SYS'], mode='lines', name='SYS', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['DIA'], mode='lines', name='DIA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['HR'], mode='lines', name='HR', line=dict(color='green')))
    fig.update_layout(title='Blood Pressure and Heart Rate Over Time',
                    xaxis_title='Time',
                    yaxis_title='Measurements',
                    legend=dict(x=0, y=0, traceorder='normal', orientation='h', yanchor='bottom'),
                    yaxis_range=[0, 140])
    # Add column information to tooltip
    fig.update_traces(hovertemplate='<b>%{y}</b><br> %{customdata[0]}',
                      customdata=df[['Note']].values)
    for i, row in df.iterrows():
        if pd.notna(row.get('Note', None)) and str(row['Note']).strip() != "":
            fig.add_annotation(
                x=row['Time'],
                y=row['HR'],
                text=str(row['Note'].split(',')[0]),
                showarrow=True,
                arrowhead=1,
                yshift=10,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.7)"
            )
    return fig

def main():
    '''Set up the Streamlit app'''
    st.title("BP Data Analysis")
    st.write("This app analyzes BP data over time.")
    df = get_data()
    df_grouped = df.groupby(('Date'))
    st.subheader("Data Overview")
    st.write(df.style.highlight_max(subset=['SYS','DIA', 'HR'], axis=0))
    # Create a line chart of BP over time
    for date, group in df_grouped:
        fig = generate_plot(group)
        fig.update_layout(title=f'Blood Pressure and Heart Rate on {date}')
        st.plotly_chart(fig) 



if __name__ == "__main__":
    main()

