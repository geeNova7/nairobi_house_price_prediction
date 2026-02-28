import pandas as pd
import joblib

def predict_price(data):
    model = joblib.load('../models/house_model.pkl')
    preprocessor = joblib.load('../models/preprocessor.pkl')

    df = pd.DataFrame([data])
    transformed = preprocessor.transform(df)

    prediction = model.predict(transformed)
    return prediction[0]

example = {
    'bedrooms':3,
    'bathrooms':2,
    'size_sqft':1800,
    'location': 'karen',
    'property type':'land'
}

print("predicted price", predict_price(example))