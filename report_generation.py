import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF

today_str = datetime.now().strftime("%Y-%m-%d")
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
filename = os.path.join(downloads_folder, f"vital_signs_report_{today_str}.pdf")


def generate_report(df, averages, filename=filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Vital Signs Report", ln=True, align='C')

    # Date range
    pdf.set_font("Arial", '', 12)
    date_range = f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
    pdf.cell(0, 10, date_range, ln=True)

    # Averages table
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Vitals At A Glimpse", ln=True)
    
    pdf.set_font("Arial", '', 12)
    for state in averages['state']:
        avg_row = averages[averages['state'] == state].iloc[0]
        pdf.cell(0, 10, f"{state}: HR={avg_row['HR']['mean']:.1f} (±{avg_row['HR']['std']:.1f}), "
                        f"SYS={avg_row['SYS']['mean']:.1f} (±{avg_row['SYS']['std']:.1f}), "
                        f"DIA={avg_row['DIA']['mean']:.1f} (±{avg_row['DIA']['std']:.1f})", ln=True)

    ## Show the full averages tables
    for vital in ["HR", "SYS", "DIA"]:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{vital} Statistics", ln=True)
        pdf.set_font("Arial", 'B', 10)
        # Table header
        pdf.cell(30, 8, "State", border=1)
        for stat in ["Mean", "Std", "Median", "Min", "Max"]:
            pdf.cell(22, 8, stat, border=1)
        pdf.ln()
        pdf.set_font("Arial", '', 10)
        # Table rows
        for idx, row in averages.iterrows():
            pdf.cell(30, 8, str(row['state'].values[0]), border=1)
            pdf.cell(22, 8, f"{row[vital]['mean']:.1f}", border=1)
            pdf.cell(22, 8, f"{row[vital]['std']:.1f}", border=1)
            pdf.cell(22, 8, f"{row[vital]['median']:.1f}", border=1)
            pdf.cell(22, 8, f"{row[vital]['min']:.1f}", border=1)
            pdf.cell(22, 8, f"{row[vital]['max']:.1f}", border=1)
            pdf.ln()
    

    return pdf.output(filename).encode("latin-1") 

