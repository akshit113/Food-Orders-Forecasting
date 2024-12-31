import numpy as np
from pandas import read_csv, get_dummies, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time

from xgboost import XGBRegressor

original_columns = ['Period_No', 'Facility_No', 'Facility_Category', 'City_Zip_Code',
                    'Operational_Region_Coverage_Area', 'Billing_Amount', 'Labelled_Price', 'Custom_Promoted',
                    'Promoted', 'Search_Promotions', 'Orders_Count', 'Course', 'Flavour_Profile']


def load_data():
    train = read_csv('./Dataset/Train.csv', header=0)
    test = read_csv('./Dataset/Test.csv', header=0)
    print(train.head(100))
    print(list(train.columns))
    print(list(test.columns))

    train['Orders_Count'] = train.pop('Orders_Count')
    return train, test


def feat_engg(X):
    region_aggregates = X.groupby('Operational_Region_Coverage_Area').agg({
        'Billing_Amount': ['mean', 'sum'],
        'Labelled_Price': ['mean', 'sum']
    }).reset_index()
    # region_aggregates = X.groupby('Operational_Region_Coverage_Area').agg({
    #     'Billing_Amount': ['mean', 'sum'],
    #     'Labelled_Price': ['mean', 'sum']
    # }).reset_index()

    region_aggregates.columns = ['Operational_Region_Coverage_Area', 'Billing_Amount__mean', 'Billing_Amount__sum',
                                 'Labelled_Price__mean', 'Labelled_Price__sum']

    # Merge aggregates back to the main dataframe
    X = X.merge(region_aggregates, on='Operational_Region_Coverage_Area', how='left')
    # Combine promotions into a single feature
    X['Total_Promotions'] = X['Custom_Promoted'] + X['Promoted'] + X['Search_Promotions']
    # Price ratio
    X['Discount_Ratio'] = X['Billing_Amount'] / X['Labelled_Price']
    # Aggregate trends by Course and Flavour_Profile
    course_flavor_trends = X.groupby(['Course', 'Flavour_Profile']).agg({
        'Billing_Amount': ['mean', 'sum'],
        'Labelled_Price': ['mean', 'sum']
    }).reset_index()

    course_flavor_trends.columns = ['Course', 'Flavour_Profile', 'Billing_Amount__mean', 'Billing_Amount__sum',
                                    'Labelled_Price__mean', 'Labelled_Price__sum']

    # Merge back to the main dataframe
    X = X.merge(course_flavor_trends, on=['Course', 'Flavour_Profile'], how='left')

    return X


if __name__ == '__main__':
    tic = time.perf_counter()
    data, pred_df = load_data()
    # perform_eda(data, pred_df)

    sampled_data = data.sample(frac=0.4, random_state=42)
    X = sampled_data.drop(columns=['Orders_Count'])
    y = sampled_data['Orders_Count']

    X = feat_engg(X)
    pred_df = feat_engg(pred_df)


    cat_cols = ['Period_No', 'Facility_No', 'Facility_Category', 'City_Zip_Code', 'Operational_Region_Coverage_Area',
                'Search_Promotions', 'Course', 'Flavour_Profile', 'Total_Promotions', 'Custom_Promoted', 'Promoted']

    num_cols = ['Billing_Amount', 'Labelled_Price', 'Billing_Amount__mean_x', 'Billing_Amount__sum_x',
                'Labelled_Price__mean_x', 'Labelled_Price__sum_x', 'Discount_Ratio', 'Billing_Amount__mean_y',
                'Billing_Amount__sum_y', 'Labelled_Price__mean_y', 'Labelled_Price__sum_y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.columns)
    print(set(X.columns).difference(set(num_cols + cat_cols)))


    def custom_combiner(feature_name, category):
        return f"{feature_name}__{category}"


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore', feature_name_combiner=custom_combiner), cat_cols)
            # Encode categorical features
        ]
    )

    # Define the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),  # Preprocessing step
        #     ('model', XGBRegressor(n_estimators=850, verbosity=2, colsample_bytree=0.75, n_jobs=9))  # Model step
        # ], verbose=True)
        ('model', RandomForestRegressor(n_estimators=10, random_state=42, verbose=3, n_jobs=9))
        # Model step
    ], verbose=True)

    print(X_train.columns)
    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Access the preprocessor step from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']

    # Get feature names for numeric features (no changes from StandardScaler)
    numeric_feature_names = preprocessor.transformers_[0][2]  # ColumnTransformer's numeric column names

    # Get feature names for categorical features (OneHotEncoder expands them)
    categorical_transformer = preprocessor.named_transformers_['cat']  # Access the OneHotEncoder
    categorical_feature_names = categorical_transformer.get_feature_names_out(cat_cols)
    print(list(categorical_feature_names))
    # Combine numeric and categorical feature names
    feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

    # Display the final list of feature names
    print("Extracted Feature Names:")
    print(feature_names)

    rf_regressor = pipeline.named_steps['model']
    feature_importances = rf_regressor.feature_importances_
    importance_df = DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    importance_df['Original_Column'] = importance_df['Feature'].str.split('__').str[0]
    # Aggregate importances by original columns
    aggregated_importances = importance_df.groupby('Original_Column')['Importance'].sum()

    print("Aggregated Feature Importances by Original Columns:")
    print(aggregated_importances)

    # Predict on the validation set
    y_val = pipeline.predict(X_test)
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_val)
    rmse = root_mean_squared_error(y_test, y_val)
    mape = mean_absolute_percentage_error(y_test, y_val)

    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Percent Error: {mape}")

    ###

    y_pred = np.round(pipeline.predict(pred_df), 0)
    df = DataFrame(y_pred, columns=['Orders_Count'])
    df.to_csv('Submission.csv', index=False)

    toc = time.perf_counter()

    print(f"Time: {toc - tic:0.4f} seconds")
    print('Program Execution Complete...')
