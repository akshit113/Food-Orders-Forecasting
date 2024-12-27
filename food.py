import numpy as np
from pandas import read_csv, get_dummies, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time

cat_cols = ['Period_No', 'Facility_No', 'Facility_Category', 'City_Zip_Code', 'Operational_Region_Coverage_Area',
            'Custom_Promoted', 'Promoted', 'Search_Promotions',
            'Course', 'Flavour_Profile']

num_cols = ['Billing_Amount', 'Labelled_Price']


def load_data():
    train = read_csv('./Dataset/Train.csv', header=0)
    test = read_csv('./Dataset/Test.csv', header=0)
    print(train.head(100))
    print(list(train.columns))
    print(list(test.columns))

    train['Orders_Count'] = train.pop('Orders_Count')
    return train, test


def perform_eda(train_df, test_df):
    for col_name in cat_cols:
        print(f'now checking ----> {col_name}')
        print(f'{len(set(train_df[col_name].values))}')
        print(f'{len(set(test_df[col_name].values))}')
        print('=====================================================================')

    print(list(train_df.columns))
    print(list(test_df.columns))

    train_df[cat_cols] = train_df[cat_cols].astype('category')
    test_df[cat_cols] = test_df[cat_cols].astype('category')
    print(train_df.dtypes)
    print(test_df.dtypes)

    train_df_encoded = get_dummies(train_df, columns=cat_cols, prefix='category', drop_first=True)
    test_df_encoded = get_dummies(test_df, columns=cat_cols, prefix='category', drop_first=True)

    return


def clean_data(train_df, test_df):
    return


if __name__ == '__main__':
    tic = time.perf_counter()
    data, pred_df = load_data()
    # perform_eda(data, pred_df)

    sampled_data = data.sample(frac=0.4, random_state=42)
    X = sampled_data.drop(columns=['Orders_Count'])
    y = sampled_data['Orders_Count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)  # Encode categorical features
        ]
    )

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocessing step
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Model step
    ], verbose=True)

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predict on the validation set
    y_val = np.floor(pipeline.predict(X_test))
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_val)
    rmse = root_mean_squared_error(y_test, y_val)
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")


    ###

    y_pred = np.round(pipeline.predict(pred_df),0)
    df = DataFrame(y_pred, columns=['Orders_Count'])
    df.to_csv('Submission.csv', index=False)

    toc = time.perf_counter()

    print(f"Time: {toc - tic:0.4f} seconds")
    print('Program Execution Complete...')
