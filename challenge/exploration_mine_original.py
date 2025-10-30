#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# ## 0. Load Data

# In[63]:


data = pd.read_csv('../data/data.csv')
data.info()


# ## 1. Data Analysis: First Sight

# ### How is the date distribuited?

# In[64]:


'''
flights_by_airline = data['OPERA'].value_counts()
plt.figure(figsize = (10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.9)
plt.title('Flights by Airline')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Airline', fontsize=12)
plt.xticks(rotation=90)
plt.show()


El error se produjo porque en las versiones nuevas de Seaborn la función barplot ya no permite pasar los valores de x y y como argumentos posicionales. La llamada original entregaba ambos sin especificar sus nombres y por eso se generó el TypeError. La sintaxis correcta exige ahora indicar los nombres de los parámetros, es decir usar x= y y= para cumplir con la nueva forma de la función.

'''

flights_by_airline = data['OPERA'].value_counts()

plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)

plt.title('Flights by Airline')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Airline', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[65]:


'''flights_by_day = data['DIA'].value_counts()
plt.figure(figsize = (10, 2))
sns.set(style = "darkgrid")
sns.barplot(flights_by_day.index, flights_by_day.values, color = 'lightblue', alpha=0.8)
plt.title('Flights by Day')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Day', fontsize=12)
plt.xticks(rotation=90)
plt.show()


El error se produjo porque en las versiones nuevas de Seaborn la función barplot ya no permite pasar los valores de x y y como argumentos posicionales. La llamada original entregaba ambos sin especificar sus nombres y por eso se generó el TypeError. La sintaxis correcta exige ahora indicar los nombres de los parámetros, es decir usar x= y y= para cumplir con la nueva forma de la función.

'''

flights_by_day = data['DIA'].value_counts()

plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(x=flights_by_day.index, y=flights_by_day.values, color='lightblue', alpha=0.8)

plt.title('Flights by Day')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Day', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[66]:


'''
flights_by_month = data['MES'].value_counts()
plt.figure(figsize = (10, 2))
sns.set(style = "darkgrid")
sns.barplot(flights_by_month.index, flights_by_month.values, color = 'lightblue', alpha=0.8)
plt.title('Flights by Month')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation=90)
plt.show()


El error se produjo porque en las versiones nuevas de Seaborn la función barplot ya no permite pasar los valores de x y y como argumentos posicionales. La llamada original entregaba ambos sin especificar sus nombres y por eso se generó el TypeError. La sintaxis correcta exige ahora indicar los nombres de los parámetros, es decir usar x= y y= para cumplir con la nueva forma de la función.

'''

flights_by_month = data['MES'].value_counts()

plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(x=flights_by_month.index, y=flights_by_month.values, color='lightblue', alpha=0.8)

plt.title('Flights by Month')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[67]:


'''
flights_by_day_in_week = data['DIANOM'].value_counts()
days = [
    flights_by_day_in_week.index[2], 
    flights_by_day_in_week.index[5], 
    flights_by_day_in_week.index[4], 
    flights_by_day_in_week.index[1], 
    flights_by_day_in_week.index[0], 
    flights_by_day_in_week.index[6], 
    flights_by_day_in_week.index[3]
]
values_by_day = [
    flights_by_day_in_week.values[2], 
    flights_by_day_in_week.values[5], 
    flights_by_day_in_week.values[4], 
    flights_by_day_in_week.values[1], 
    flights_by_day_in_week.values[0], 
    flights_by_day_in_week.values[6], 
    flights_by_day_in_week.values[3]
]
plt.figure(figsize = (10, 2))
sns.set(style="darkgrid")
sns.barplot(days, values_by_day, color = 'lightblue', alpha=0.8)
plt.title('Flights by Day in Week')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Day in Week', fontsize=12)
plt.xticks(rotation=90)
plt.show()

El error se generó porque la función barplot de Seaborn ya no admite argumentos posicionales para los valores de x y y. En este caso se pasaron las listas days y values_by_day sin nombrar los parámetros, lo que provocó el TypeError. La forma correcta es indicar los nombres de los argumentos usando x=days y y=values_by_day para cumplir con la nueva sintaxis requerida por las versiones recientes de la librería.
'''

flights_by_day_in_week = data['DIANOM'].value_counts()

days = [
    flights_by_day_in_week.index[2], 
    flights_by_day_in_week.index[5], 
    flights_by_day_in_week.index[4], 
    flights_by_day_in_week.index[1], 
    flights_by_day_in_week.index[0], 
    flights_by_day_in_week.index[6], 
    flights_by_day_in_week.index[3]
]

values_by_day = [
    flights_by_day_in_week.values[2], 
    flights_by_day_in_week.values[5], 
    flights_by_day_in_week.values[4], 
    flights_by_day_in_week.values[1], 
    flights_by_day_in_week.values[0], 
    flights_by_day_in_week.values[6], 
    flights_by_day_in_week.values[3]
]

plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(x=days, y=values_by_day, color='lightblue', alpha=0.8)

plt.title('Flights by Day in Week')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Day in Week', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[68]:


'''flights_by_type = data['TIPOVUELO'].value_counts()
sns.set(style="darkgrid")
plt.figure(figsize = (10, 2))
sns.barplot(flights_by_type.index, flights_by_type.values, alpha=0.9)
plt.title('Flights by Type')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Type', fontsize=12)
plt.show()

El error ocurrió porque en las versiones recientes de Seaborn la función barplot() ya no acepta argumentos posicionales. En el código original se pasaron los valores del eje x y del eje y sin nombrarlos, lo que provocó el TypeError. La sintaxis actual exige indicar los nombres de los parámetros (x= y y=) para definir correctamente las variables de cada eje.
'''


flights_by_type = data['TIPOVUELO'].value_counts()

sns.set(style="darkgrid")
plt.figure(figsize=(10, 2))
sns.barplot(x=flights_by_type.index, y=flights_by_type.values, alpha=0.9)

plt.title('Flights by Type')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Type', fontsize=12)
plt.show()


# In[69]:


'''flight_by_destination = data['SIGLADES'].value_counts()
plt.figure(figsize = (10, 2))
sns.set(style="darkgrid")
sns.barplot(flight_by_destination.index, flight_by_destination.values, color = 'lightblue', alpha=0.8)
plt.title('Flight by Destination')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Destination', fontsize=12)
plt.xticks(rotation=90)
plt.show()

El error se produjo porque la función barplot() de Seaborn ya no acepta argumentos posicionales. En este caso se pasaron los valores del eje x y del eje y sin especificar sus nombres, lo que generó el TypeError. La forma correcta es usar los parámetros con nombre x= y y= para cumplir con la sintaxis actual de la librería.

'''

flight_by_destination = data['SIGLADES'].value_counts()

plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(x=flight_by_destination.index, y=flight_by_destination.values, color='lightblue', alpha=0.8)

plt.title('Flight by Destination')
plt.ylabel('Flights', fontsize=12)
plt.xlabel('Destination', fontsize=12)
plt.xticks(rotation=90)
plt.show()



# ## 2. Features Generation

# ### 2.a. Period of Day

# In[70]:


from datetime import datetime

def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'


# In[71]:


data['period_day'] = data['Fecha-I'].apply(get_period_day)


# ### 2.b. High Season

# In[72]:


from datetime import datetime

def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
    if ((fecha >= range1_min and fecha <= range1_max) or 
        (fecha >= range2_min and fecha <= range2_max) or 
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0


# In[73]:


data['high_season'] = data['Fecha-I'].apply(is_high_season)


# ### 2.c. Difference in Minutes

# In[74]:


def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff


# In[75]:


data['min_diff'] = data.apply(get_min_diff, axis = 1)


# ### 2.d. Delay

# In[76]:


threshold_in_minutes = 15
data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)


# In[77]:


data.columns


# ## 3. Data Analysis: Second Sight

# ### How is the delay rate across columns?

# In[78]:


def get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()
    
    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0
            
    return pd.DataFrame.from_dict(data = rates, orient = 'index', columns = ['Tasa (%)'])


# In[79]:


'''
destination_rate = get_rate_from_column(data, 'SIGLADES')
destination_rate_values = data['SIGLADES'].value_counts().index
plt.figure(figsize = (20,5))
sns.set(style="darkgrid")
sns.barplot(destination_rate_values, destination_rate['Tasa (%)'], alpha = 0.75)
plt.title('Delay Rate by Destination')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Destination', fontsize=12)
plt.xticks(rotation=90)
plt.show()


El error se generó porque la función barplot() de Seaborn ya no admite argumentos posicionales. En el código original se pasaron los valores del eje x y del eje y sin nombrarlos, lo que provocó el TypeError. La nueva sintaxis requiere usar los parámetros con nombre (x= y y=) para especificar cada eje.

'''


destination_rate = get_rate_from_column(data, 'SIGLADES')
destination_rate_values = data['SIGLADES'].value_counts().index

plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(x=destination_rate_values, y=destination_rate['Tasa (%)'], alpha=0.75)

plt.title('Delay Rate by Destination')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Destination', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[80]:


'''
airlines_rate = get_rate_from_column(data, 'OPERA')
airlines_rate_values = data['OPERA'].value_counts().index
plt.figure(figsize = (20,5))
sns.set(style="darkgrid")
sns.barplot(airlines_rate_values, airlines_rate['Tasa (%)'], alpha = 0.75)
plt.title('Delay Rate by Airline')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Airline', fontsize=12)
plt.xticks(rotation=90)
plt.show()

El error ocurrió porque la función barplot() de Seaborn ya no acepta argumentos posicionales. En este caso se pasaron los valores de los ejes sin indicar sus nombres, lo que generó el TypeError. La forma correcta en las versiones actuales de la librería es usar los parámetros con nombre (x= y y=).
'''


airlines_rate = get_rate_from_column(data, 'OPERA')
airlines_rate_values = data['OPERA'].value_counts().index

plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(x=airlines_rate_values, y=airlines_rate['Tasa (%)'], alpha=0.75)

plt.title('Delay Rate by Airline')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Airline', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# In[81]:


'''
month_rate = get_rate_from_column(data, 'MES')
month_rate_value = data['MES'].value_counts().index
plt.figure(figsize = (20,5))
sns.set(style="darkgrid")
sns.barplot(month_rate_value, month_rate['Tasa (%)'], color = 'blue', alpha = 0.75)
plt.title('Delay Rate by Month')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0,10)
plt.show()

El error se debe a que Seaborn ya no acepta argumentos posicionales en la función barplot(). En las versiones nuevas (≥0.12), es obligatorio indicar los parámetros con nombre, es decir, usar x= y y=.
'''

month_rate = get_rate_from_column(data, 'MES')
month_rate_value = data['MES'].value_counts().index

plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(x=month_rate_value, y=month_rate['Tasa (%)'], color='blue', alpha=0.75)

plt.title('Delay Rate by Month')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Month', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0, 10)
plt.show()


# In[82]:


'''
days_rate = get_rate_from_column(data, 'DIANOM')
days_rate_value = data['DIANOM'].value_counts().index

sns.set(style="darkgrid")
plt.figure(figsize = (20, 5))
sns.barplot(days_rate_value, days_rate['Tasa (%)'], color = 'blue', alpha = 0.75)
plt.title('Delay Rate by Day')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Days', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0,7)
plt.show()

El error ocurrió porque en las versiones nuevas de Seaborn la función barplot() ya no acepta argumentos posicionales. En el código se pasaron los valores de los ejes sin indicar sus nombres, lo que provocó el TypeError. Ahora es obligatorio usar los parámetros con nombre x= y y=.
'''


days_rate = get_rate_from_column(data, 'DIANOM')
days_rate_value = data['DIANOM'].value_counts().index

sns.set(style="darkgrid")
plt.figure(figsize=(20, 5))
sns.barplot(x=days_rate_value, y=days_rate['Tasa (%)'], color='blue', alpha=0.75)

plt.title('Delay Rate by Day')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Days', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0, 7)
plt.show()


# In[83]:


'''
high_season_rate = get_rate_from_column(data, 'high_season')
high_season_rate_values = data['high_season'].value_counts().index

plt.figure(figsize = (5, 2))
sns.set(style="darkgrid")
sns.barplot(["no", "yes"], high_season_rate['Tasa (%)'])
plt.title('Delay Rate by Season')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('High Season', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0,6)
plt.show()

El error se debe a que, en las versiones recientes de Seaborn, la función barplot() ya no acepta argumentos posicionales. En el código se pasaron las listas de valores sin nombrarlas, lo que generó el TypeError.
La forma correcta es usar los parámetros con nombre x= y y=.
'''

high_season_rate = get_rate_from_column(data, 'high_season')
high_season_rate_values = data['high_season'].value_counts().index

plt.figure(figsize=(5, 2))
sns.set(style="darkgrid")
sns.barplot(x=["no", "yes"], y=high_season_rate['Tasa (%)'])

plt.title('Delay Rate by Season')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('High Season', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0, 6)
plt.show()


# In[84]:


'''
flight_type_rate = get_rate_from_column(data, 'TIPOVUELO')
flight_type_rate_values = data['TIPOVUELO'].value_counts().index
plt.figure(figsize = (5, 2))
sns.set(style="darkgrid")
sns.barplot(flight_type_rate_values, flight_type_rate['Tasa (%)'])
plt.title('Delay Rate by Flight Type')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Flight Type', fontsize=12)
plt.ylim(0,7)
plt.show()

El error se produjo porque en las versiones actuales de Seaborn la función barplot() ya no admite argumentos posicionales. En el código se pasaron los valores de los ejes sin indicar los nombres de los parámetros, lo que generó el TypeError. La sintaxis correcta ahora requiere usar x= y y=.
'''

flight_type_rate = get_rate_from_column(data, 'TIPOVUELO')
flight_type_rate_values = data['TIPOVUELO'].value_counts().index

plt.figure(figsize=(5, 2))
sns.set(style="darkgrid")
sns.barplot(x=flight_type_rate_values, y=flight_type_rate['Tasa (%)'])

plt.title('Delay Rate by Flight Type')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Flight Type', fontsize=12)
plt.ylim(0, 7)
plt.show()


# In[85]:


'''
period_day_rate = get_rate_from_column(data, 'period_day')
period_day_rate_values = data['period_day'].value_counts().index
plt.figure(figsize = (5, 2))
sns.set(style="darkgrid")
sns.barplot(period_day_rate_values, period_day_rate['Tasa (%)'])
plt.title('Delay Rate by Period of Day')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Period', fontsize=12)
plt.ylim(3,7)
plt.show()


El error ocurrió porque en las versiones recientes de Seaborn la función barplot() ya no permite pasar los valores de los ejes como argumentos posicionales. En el código se entregaron directamente period_day_rate_values y period_day_rate['Tasa (%)'] sin indicar sus nombres, lo que generó el TypeError. La forma correcta es usar x= y y=.


'''

period_day_rate = get_rate_from_column(data, 'period_day')
period_day_rate_values = data['period_day'].value_counts().index

plt.figure(figsize=(5, 2))
sns.set(style="darkgrid")
sns.barplot(x=period_day_rate_values, y=period_day_rate['Tasa (%)'])

plt.title('Delay Rate by Period of Day')
plt.ylabel('Delay Rate [%]', fontsize=12)
plt.xlabel('Period', fontsize=12)
plt.ylim(3, 7)
plt.show()



# ## 4. Training

# ### 4.a. Data Split (Training and Validation)

# In[86]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report


# In[87]:


training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)


# In[88]:


features = pd.concat([
    pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
    pd.get_dummies(data['MES'], prefix = 'MES')], 
    axis = 1
)
target = data['delay']


# In[89]:


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)


# In[90]:


print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")


# In[91]:


y_train.value_counts('%')*100


# In[92]:


y_test.value_counts('%')*100


# ### 4.b. Model Selection

# #### 4.b.i. XGBoost

# In[93]:


import xgboost as xgb
from xgboost import plot_importance


# In[94]:


xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
xgb_model.fit(x_train, y_train)


# In[95]:


xgboost_y_preds = xgb_model.predict(x_test)
xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]


# In[96]:


confusion_matrix(y_test, xgboost_y_preds)


# In[97]:


print(classification_report(y_test, xgboost_y_preds))


# #### 4.b.ii. Logistic Regression

# In[98]:


from sklearn.linear_model import LogisticRegression


# In[99]:


reg_model = LogisticRegression()
reg_model.fit(x_train, y_train)


# In[100]:


reg_y_preds = reg_model.predict(x_test)


# In[101]:


confusion_matrix(y_test, reg_y_preds)


# In[102]:


print(classification_report(y_test, reg_y_preds))


# ## 5. Data Analysis: Third Sight

# ### Feature Importance

# In[103]:


plt.figure(figsize = (10,5))
plot_importance(xgb_model)


# In[104]:


top_10_features = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]


# ### Data Balance

# In[105]:


n_y0 = len(y_train[y_train == 0])
n_y1 = len(y_train[y_train == 1])
scale = n_y0/n_y1
print(scale)


# ## 6. Training with Improvement

# ### 6.a. Data Split

# In[106]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(features[top_10_features], target, test_size = 0.33, random_state = 42)


# ### 6.b. Model Selection

# #### 6.b.i. XGBoost with Feature Importance and with Balance

# In[107]:


xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
xgb_model_2.fit(x_train2, y_train2)


# In[108]:


xgboost_y_preds_2 = xgb_model_2.predict(x_test2)


# In[109]:


confusion_matrix(y_test2, xgboost_y_preds_2)


# In[110]:


print(classification_report(y_test2, xgboost_y_preds_2))


# #### 6.b.ii. XGBoost with Feature Importance but without Balance

# In[111]:


xgb_model_3 = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
xgb_model_3.fit(x_train2, y_train2)


# In[112]:


xgboost_y_preds_3 = xgb_model_3.predict(x_test2)


# In[113]:


confusion_matrix(y_test2, xgboost_y_preds_3)


# In[114]:


print(classification_report(y_test2, xgboost_y_preds_3))


# #### 6.b.iii. Logistic Regression with Feature Importante and with Balance

# In[115]:


reg_model_2 = LogisticRegression(class_weight={1: n_y0/len(y_train), 0: n_y1/len(y_train)})
reg_model_2.fit(x_train2, y_train2)


# In[116]:


reg_y_preds_2 = reg_model_2.predict(x_test2)


# In[117]:


confusion_matrix(y_test2, reg_y_preds_2)


# In[118]:


print(classification_report(y_test2, reg_y_preds_2))


# #### 6.b.iv. Logistic Regression with Feature Importante but without Balance

# In[119]:


reg_model_3 = LogisticRegression()
reg_model_3.fit(x_train2, y_train2)


# In[120]:


reg_y_preds_3 = reg_model_3.predict(x_test2)


# In[121]:


confusion_matrix(y_test2, reg_y_preds_3)


# In[122]:


print(classification_report(y_test2, reg_y_preds_3))


# ## 7. Data Science Conclusions

# By looking at the results of the 6 trained models, it can be determined:
# - There is no noticeable difference in results between XGBoost and LogisticRegression.
# - Does not decrease the performance of the model by reducing the features to the 10 most important.
# - Improves the model's performance when balancing classes, since it increases the recall of class "1".

# **With this, the model to be productive must be the one that is trained with the top 10 features and class balancing, but which one?**

# 
