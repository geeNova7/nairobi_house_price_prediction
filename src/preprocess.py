import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_data(path='../data/nairobi_property_prices.csv'):
    df = pd.read_csv(path)

    #drop any ID column
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Remove everything except digits
    df['Price'] = (
        df['Price']
        .astype(str)
        .str.replace(r'[^0-9]', '', regex=True)
    )

    # Convert price values to numeric
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    #separate target
    x = df.drop('Price', axis=1)
    y = df['Price']

    #identity features
    categorical = x.select_dtypes(include='object').columns
    numerical = x.select_dtypes(include=['int64', 'float64']).columns

    #numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical),
        ('cat', cat_pipeline, categorical)
    ])

    x_processed = preprocessor.fit_transform(x)

    return x_processed, y, preprocessor