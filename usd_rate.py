import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

dataframe = pandas.read_excel("usd_rate.xlsx")
rate = dataframe.curs

past = 7 # Для обучения будем брать данные за 7 дней из прошлого
# Будем пытаться на их основе спрогнозировать курс на завтра

def models(prediction, y_test):
    plt.plot(prediction, label="Prediction")
    plt.plot(list(y_test), label="Real")
    plt.legend()
    mae = mean_absolute_error(prediction, y_test)
    print(f"MAE = {mae}")


length = len(rate)
count = length - past

past_days = []
current_day = []

for day in range(past, length):
    slc_x = list(rate[(day-past):day])
    past_days.append(slc_x) # rate[4:33]
    slc_y = rate[day]
    current_day.append(slc_y) # rate[33]


past_columns = []
for i in range(past):
    past_columns.append(f"past_{i}")

x = pandas.DataFrame(data=past_days, columns=past_columns)
# То, на основе чего мы делаем предсказание

y = pandas.Series(current_day, name='target')
# То, что мы пытаемся предсказать

# Обучающая выборка, "Учебник"
x_train = x[:-10]
y_train = y[:-10]

# Тестовая выборка, "Экзамен"
x_test = x[-10:]
y_test = y[-10:]

regressor = LinearRegression()
# Модель линейной регрессии

grid = {}
GS = GridSearchCV(regressor, grid, cv = 10, scoring = 'neg_mean_absolute_error')
GS.fit(x_train, y_train)
best_model = GS.best_estimator_
prediction = best_model.predict(x_test)
models(prediction, y_test)

print("Введите курс доллара за эту неделю: ")
# Тестовая выборка, "Экзамен"
x_test = [[]]
for i in range(7):
    day = float(input(str(i+1) + ' день: '))
    x_test[0].append(day)


x_test = pandas.DataFrame(data=x_test, columns=past_columns)
prediction = best_model.predict(x_test)
print("$$$ Доллар будет стоить: " + str(prediction[0]) + " рублей" )