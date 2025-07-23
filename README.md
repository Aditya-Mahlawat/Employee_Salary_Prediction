# Employee Salary Prediction

## Overview
This project predicts whether an employee's salary is above or below $50,000 per year based on various attributes such as age, education level, occupation, and working hours. It uses machine learning to classify employees into two salary categories: >50K or ≤50K.

## Features
- Interactive web interface built with Streamlit
- Single prediction for individual employee data
- Batch prediction for multiple employees via CSV upload
- Trained using K-Nearest Neighbors (KNN) algorithm

## Dataset
The model is trained on the Adult Census Income dataset (also known as the "Census Income" dataset), which contains demographic information about individuals along with their income level. The dataset includes features such as:

- Age
- Education level
- Occupation
- Hours worked per week
- Experience
- And more

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install streamlit pandas scikit-learn joblib
```

## Usage

### Running the App
To run the application, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start a local server and open the application in your default web browser.

### Making Predictions
1. **Individual Prediction**:
   - Enter employee details in the sidebar
   - Click the "Predict Salary Class" button
   - View the prediction result

2. **Batch Prediction**:
   - Prepare a CSV file with the same columns as the input form
   - Upload the CSV file in the batch prediction section
   - Download the results with predictions added

## Model Information
The salary prediction model uses the K-Nearest Neighbors (KNN) algorithm, which classifies new data points based on the majority class of their nearest neighbors in the training dataset.

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Accuracy**: Approximately 82% on test data
- **Features used**: age, education, occupation, hours-per-week, experience

## Project Structure
- `app.py`: Streamlit web application
- `best_model.pkl`: Trained KNN model
- `adult 3.csv`: Training dataset
- `knn_adult_csv updated.ipynb`: Jupyter notebook with model training code

## Future Improvements
- Add more visualization of prediction results
- Implement feature importance analysis
- Try different machine learning algorithms for comparison
- Add user authentication for secure access

## License
This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Contact
For questions or feedback, please open an issue in this repository.