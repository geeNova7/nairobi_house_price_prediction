from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from preprocess import preprocess_data

#load data
x, y, preprocessor = preprocess_data()

train_x, test_x, train_y, test_y = train_test_split(
    x,y, test_size=0.2, random_state=33)

house_model = RandomForestRegressor(
    n_estimators=200,
    random_state=33
)

house_model.fit(train_x, train_y)
y_pred = house_model.predict(test_x)
y_val_mae = mean_absolute_error(y_pred,test_y)
r2 = r2_score(y_pred, test_y)

print("MAE:",y_val_mae)
print("R2 score", r2)

#save model and preprocessor
joblib.dump(house_model,'../models/house_model.pkl')
joblib.dump(preprocessor, '../models/preprocessor.pkl')