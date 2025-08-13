# Employee Productivity Predictor

> A **web-based Employee Productivity Prediction System** built using a **Linear Regression** algorithm and deployed with Flask.
> Predicts an employee's potential productivity score (out of 100) based on metrics like **Work Hours**, **Break Time**, and **Tasks Completed**.

-----

## Overview

This project uses a **Linear Regression machine learning model** to predict an employee's productivity score.
The **Flask web app** allows users to:

  - Enter an employee's work metrics
  - Get an instant productivity score prediction
  - View the simple, clean result on a web page

-----

## Project Structure

```
productivity-predictor/
│
├── app.py # Flask web application
├── train_model.py # Script to generate a dataset, train, and save the Linear Regression model
├── dataset/
│ └── productivity_data.csv # Dataset used to train the model
│
├── model/
│ ├── lr_model.pkl # Serialized Linear Regression model
│ └── scaler.pkl # Serialized scaler for preprocessing
│
├── templates/
│ └── index.html # Input form and result page UI
│
└── README.md # Documentation
```

-----

## How It Works

### 1 Model Training (`train_model.py`)

  - Generates a synthetic dataset for employee productivity and saves it to `productivity_data.csv`.
  - Scales features (`Work_Hours`, `Break_Time`, `Tasks_Completed`) using `StandardScaler`.
  - Splits the data into training and testing sets.
  - Trains a **Linear Regression** model.
  - Saves the trained model (`lr_model.pkl`) and the scaler (`scaler.pkl`) for use in the web application.

### 2 Web Application (`app.py`)

  - Loads the pre-trained model and scaler.
  - Accepts **user inputs** (Work Hours, Break Time, Tasks Completed) from a web form.
  - Scales the user input data using the saved `scaler.pkl`.
  - Predicts the productivity score using `lr_model.pkl`.
  - Displays the predicted score on the `index.html` page.

-----

## Example Visuals

### Sample Dataset

Example rows from `dataset/productivity_data.csv`:

-----

### Web App Input Form
<img width="481" height="415" alt="image" src="https://github.com/user-attachments/assets/a3018449-47cf-4d36-8d34-b5d704742bdb" />

-----

### Prediction Result Page
<img width="460" height="459" alt="image" src="https://github.com/user-attachments/assets/24c9d039-0419-4541-a647-225f31ef6840" />


-----

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/nithinskumar866/Linear_regression-model.git
cd productivity-predictor

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the web application
python app.py
```

Visit **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

-----

## Sample Dataset

Example row from `dataset/productivity_data.csv`:

```csv
Work_Hours,Break_Time,Tasks_Completed,Productivity
8.745071,1.075148,3.34201,100.0
7.792604,1.103934,3.879638,100.0
8.971533,0.795993,6.494587,100.0
...
```

-----

## Features

  - Predict productivity using **Linear Regression**.
  - Accepts **Work Hours, Break Time, and Tasks Completed** as input.
  - Simple Flask-based web interface.
  - Fast predictions from a pre-trained model.
  - Scalable and easy to integrate into other systems.

-----

## Future Enhancements

  * Add user authentication and profiles.
  * Track individual employee productivity over time.
  * Incorporate more features (e.g., project complexity, team collaboration).
  * Provide productivity recommendations based on the prediction.
  * Mobile-friendly design.
