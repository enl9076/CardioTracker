from datetime import datetime
import pandas as pd
from fpdf import FPDF

def generate_report(df, averages, filename="report.pdf"):
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
    pdf.cell(0, 10, "Average Vital Signs by Position", ln=True)
    
    pdf.set_font("Arial", '', 12)
    for state in averages['state']:
        avg_row = averages[averages['state'] == state].iloc[0]
        pdf.cell(0, 10, f"{state}: HR={avg_row['HR']['mean']:.1f} (±{avg_row['HR']['std']:.1f}), "
                        f"SYS={avg_row['SYS']['mean']:.1f} (±{avg_row['SYS']['std']:.1f}), "
                        f"DIA={avg_row['DIA']['mean']:.1f} (±{avg_row['DIA']['std']:.1f})", ln=True)

    # Save the PDF
    return pdf.output(filename).encode("latin-1") 

