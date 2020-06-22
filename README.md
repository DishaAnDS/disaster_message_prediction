# Disaster Response Pipeline Project

### Summary
This project is to use the machine learning and natural language processing technology to analyze the text messages that were sent during disaster events and categorize these events that will help appropriate disaster relief agencies to take actions.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
<<<<<<< HEAD
https://view6914b2f4-3001.udacity-student-workspaces.com/
=======

>>>>>>> a8ea1c61a378f59314b0bd3d934d72bfb2af15bf
