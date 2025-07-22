import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

def generate_hr_plots(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['HR'], mode='lines', name='HR', line=dict(color='green')))
    fig.update_layout(title='Heart Rate Over Time',
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

def generate_bp_plots(df):
    fig = go.Figure()
    # Plot SYS, DIA, and PULSE in one plot
    fig.add_trace(go.Scatter(x=df['Time'], y=df['SYS'], mode='lines', name='SYS', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['DIA'], mode='lines', name='DIA', line=dict(color='red')))
    fig.update_layout(title='Blood Pressure Over Time',
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
                y=row['SYS'],
                text=str(row['Note'].split(',')[0]),
                showarrow=True,
                arrowhead=1,
                yshift=10,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.7)"
            )
    return fig

def generate_pairplot(df):
    sns.pairplot(df, vars=['HR', 'SYS', 'DIA'], hue='state', diag_kind='kde', palette='Set2')
    plt.suptitle("Pairplot of HR, SYS, DIA by State", y=1.02)
    plt.show()
    
def generate_density_plot(df):
    sns.kdeplot(data=df, x='HR', hue='state', fill=True, common_norm=False, palette='Set2')
    plt.title("Density Plot of Heart Rate by State")
    plt.show()
    return plt.gcf()  # Return the current figure for Streamlit compatibility