import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.externals import joblib 


def load_data(database_filepath):
   """
    The function is for reading the sql .db file to a pandas dataframe
    Parameters:
        database_filepath(str) : the location of the database file.
    Returns:
        X (numpy array): input variables for the model.
        Y (numpy array): output variable for the model.
        categories_names (list): list of the names that are the targeted classes.
    """
    sql_engine = "sqlite:///" + database_filepath
    table_name = "DisasterResponse"
    engine = create_engine(sql_engine)
    df = pd.read_sql("SELECT * FROM {}".format(table_name), engine)
    Y = df.iloc[:, 5:].values
    X = df['message'].values
    categories_names = list(df.columns[5:])
    return X,Y, categories_names 


def tokenize(text):
     """
    The function is for tokenizing the given text.
    Parameters:
        text(str) : the text message needs to be tokenized
    Returns:
        clean_tokens (list): list of tokenized text message.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    The building model pipeling
    Returns:
        cv (model): trained model.
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ("multi-clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        "multi-clf__estimator__max_depth": [None, 5,10,20],
        'multi-clf__estimator__n_estimators': [50, 100],
        #'vect__ngram_range': ((1,1), (1,2))
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The building model pipeling
    Returns:
        cv (model): trained model.
    """
    for i in range(len(category_names)):
        y_pred = model.predict(X_test)
        print(classification_report(Y_test.transpose()[i], y_pred.transpose()[i]))


def save_model(model, model_filepath):
    """
    Save the trained model.
    Parameters:
        model (trained model): the trained model from training.
        model_filepath (str): the location that the model is going to be saved.
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()