import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

file_path = "E:\_CRED Lab\Codes\Python\Glass Properties\Supplementary-data.csv"  # Enter the dataset path
glass_data = pd.read_csv(file_path)
print(glass_data.columns)

y = glass_data['QSiO2 (moles SiO2/cm2/s)']
#print(y.shape)
glass_features = ['Na2O', 'Al2O3', 'SiO2', 'Initial pH']
X = glass_data[glass_features]
#print(X.describe())
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
print(val_y.shape)

glass_model = DecisionTreeRegressor(random_state = 1)
glass_model.fit(train_X,train_y)
# predicted_price = glass_model.predict(X)
# print(mean_absolute_error(y, predicted_price))

val_predictions = glass_model.predict(val_X)
print("MAE is ", mean_absolute_error(val_y, val_predictions))
r2 = r2_score(val_y, val_predictions)
print("The r2 is:", r2)
for i in val_y:
    print(i)

for i in val_predictions:
    print(i)

# print(val_predictions, '\n', val_y)
plt.plot(val_y, val_predictions, 'o')
plt.show()
