# Disaster Response Pipeline Project

### Table of Contents

* [Summary](#Summary)
* [Files-Expanation](#Files-Explanation)
* [Instructions](#Instruction)
* [License](#License)

#### Summary
This project is to use the machine learning and natural language processing technology to analyze the text messages that were sent during disaster events and categorize these events that will help appropriate disaster relief agencies to take actions. The corresponding model used in this project is random forest classifier. The UI is created by using the Flask framework.

#### Files-Explanation
1. Folder Structure <br/>
* app/ <br/>
		* templates/ <br/>
        		* go.html <br/>
                * master.html <br/>
		* run.py <br/>
 * data/ <br/>
 		* DisasterResponse.db <br/>
        * disaster_categories.csv <br/>
        * disaster_messages.csv <br/>
        * process_data.py <br/>
* models/ <br/>
        * classifier.pkl <br/>
        * train_classifier.py <br/>
* README.md <br>
2. File Main Function
`process_data.py`: read, clean and combien the .csv datasets. This is going to create the database `DisasterResponse.db`. <br/>
`run.py`: this will run the Flask app. <br/>
`templates/`: this folder provides the .html files that will defign corresponding setting and configuration of the Flask app. <br/>
`train_classifier.py`: this file will load the data, train the model. The trained model is going to be stored as `classifier.pkl`. <br/>

#### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
https://view6914b2f4-3001.udacity-student-workspaces.com/


#### License
The dataset and code template are provided by Udacity.

