# 📈 Blood Pressure & Heart Rate Data Analyzer (Streamlit App)  
A powerful and interactive Streamlit application designed to analyze and visualize your blood pressure and heart rate data. Gain insights into your cardiovascular health trends with intuitive charts and statistics.

## ✨ Features  
* **Interactive Data Upload**: Easily upload your blood pressure and heart rate data in CSV format.

* **Data Overview**: View key statistics such as average systolic, diastolic, and heart rate, along with min/max values.

* **Trend Visualization**: Visualize blood pressure (systolic and diastolic) and heart rate trends over time using interactive line charts.

* **Categorization & Tagging**: Categorize readings by context or symptom and annotate plots with context tags (e.g., sitting versus standing).

* **Distribution Plots**: Understand the distribution of your readings with histograms or density plots.

* **Customizable Filtering**: Filter data by date range or specific categories for focused analysis.

* **Responsive UI**: A clean and responsive user interface built with Streamlit, accessible on various devices.

## 💡 Usage
Once the app is running:

1. **Upload your data**: Use the file uploader widget on the sidebar to select your CSV file containing blood pressure and heart rate data.

2. **Explore visualizations**: Navigate through the different sections to view trends, distributions, and categorized analyses of your health data.

3. **Apply filters**: Use the date range selector or other filters to narrow down your analysis.

## 📊 Data Format
The application expects a CSV file with at least the following columns:

* `Date`: (e.g., YYYY-MM-DD or DD-MM-YYYY)

* `Time`: (e.g., HH:MM or HH:MM:SS)

* `SYS(mmHg)`: Systolic blood pressure (Integer, mmHg)

* `DIA(mmHg)`: Diastolic blood pressure (Integer, mmHg)

* `Pulse(Beats/Min)`: (Integer, BPM)

If you want the timeline graphs annotated with symptoms/context during the readings, the data file you upload will also need a 'Note' column (works best if there is a single word for each entry).

## 🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature/your-feature-name`).

3. Make your changes.

4. Commit your changes (`git commit -m 'Add new feature'`).

5. Push to the branch (`git push origin feature/your-feature-name`).

6. Open a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
