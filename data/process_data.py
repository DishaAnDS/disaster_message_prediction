import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    The function to read message and categories data and combine them to one dataframe.
    Parameters:
        messages_filepath (str): The path for disaster messages. This should be csv file.
        categories_filepath (str): The path for categories messages. This should be csv file.
    Returns:
        df (pandas dataframe): combined data.
    """
    # Read data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the data by common key "id"
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner')
    return df


def clean_data(df):
    """
    The function is to clean the data.
    Parameters:
        df (pandas dataframe): loaded data from load_data function.
    Returns:
        df (pandas dataframe): cleaned version of the data.
    """
    # Create a dataframe of the 36 individual category columns
    cate_df = df['categories'].str.split(";", expand = True)
    row = cate_df.head(1)
    category_colnames = [i.split("-")[0] for i in row.values.tolist()[0]]
    cate_df.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in cate_df:
        cate_df[column] = [i[-1:] for i in  cate_df[column]]
        cate_df[column] =  cate_df[column].astype(float)
    df = pd.merge(df, cate_df, left_index=True, right_index=True, how='inner')
    
    # Drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    This function save the pandas dataframe to sql database.
    Parameters:
        df (pandas dataframe): cleaned dataframe from the function clean_data.
        database_filename (str): the name of the databse.
    """
    engine = create_engine('sqlite:///' + database_filename)
    table_name = 'DisasterResponse'
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()