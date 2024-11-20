from pandas import read_csv


def load_data():
    train = read_csv('./Dataset/Train.csv', header=0)
    test = read_csv('./Dataset/Test.csv', header=0)
    print(train.head(100))
    print(list(train.columns))
    print(list(test.columns))

    train['Orders_Count'] = train.pop('Orders_Count')
    return train, test


def perform_eda(train_df, test_df):
    """EDA Results:
    1. Categorical Columns:
        Facility_Category - c1,c2,c3



    """
    print('getting unique facility category..')
    print(set(train_df['Facility_Category'].values))
    print(set(test_df['Facility_Category'].values))
    print('=====================================================================')

    print('check zip code info..')
    print((set(train_df['City_Zip_Code'].values)))
    print((set(test_df['City_Zip_Code'].values)))
    print('There are 8 distinct zipcodes. Need to replace 0 with a dummy zipcode, 123')
    print('=====================================================================')

    print('now checking Operational_Region_Coverage_Area')
    print(train_df.isnull().sum())
    print(train_df.isin([0]).sum())

    return


def clean_data(train_df, test_df):
    return


if __name__ == '__main__':
    train_df, test_df = load_data()
    perform_eda(train_df, test_df)

    print('Program Execution Complete...')
