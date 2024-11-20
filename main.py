from pandas import DataFrame, read_csv


def load_data():
    train_df = read_csv('./Dataset/Train.csv', header=0)
    test_df = read_csv('./Dataset/Test.csv', header=0)
    print(list(train_df.columns))
    print(list(test_df.columns))
    train_df['Orders_Count'] = train_df.pop('Orders_Count')
    return train_df, test_df


def perform_eda(train_df,test_df):
    pass

def clean_data(train_df,test_df):

    return





if __name__ == '__main__':
    train_df, test_df = load_data()

    print('Program Execution Complete...')
