import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

file_path = "E:\_CRED Lab\Codes\Python\Glass Properties\Supplementary-data.csv"
glass_data = pd.read_csv(file_path)
print(glass_data.columns)

y = glass_data['QSiO2 (moles SiO2/cm2/s)']
#print(y.shape)
glass_features = ['Na2O', 'Al2O3', 'SiO2', 'Initial pH']
X = glass_data[glass_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
# print(X.describe().transpose)

glass_model = MLPRegressor(hidden_layer_sizes=(6,4),activation= "tanh",solver= "adam", max_iter= 50000, random_state=1)
glass_model.fit(train_X, train_y)
val_predictions = glass_model.predict(val_X)
print("MAE is", mean_absolute_error(val_y, val_predictions))
r2 = r2_score(val_y, val_predictions)
print("The r2 is: ", r2)
# for i in val_y:
#     print(i)
#

# for i in val_predictions:
#     print(i)
plt.plot(val_y, val_predictions, 'o')
plt.show()
