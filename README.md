# 🎓 Student Performance Prediction App

This is a Streamlit web app that predicts whether a student will pass or fail based on their personal and academic attributes.

## 📦 Features
- Inputs for age, study time, absences, failures, social life, and more
- Predicts student performance using a trained Random Forest model
- Web-based interface built with Streamlit

## 🚀 How to Use

1. Place `student-mat.csv` from [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) in the same directory.
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Run the web app:
   ```bash
   streamlit run app.py
   ```

Enjoy predicting! 🎉