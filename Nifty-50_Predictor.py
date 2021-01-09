# Importing project dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import streamlit as st
import time
import pickle

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Nifty-50 Value Predictor')
st.header('Predicts Stock price data of the fifty stocks in NIFTY-50 index from NSE India.')

# Data preprocessing

data = pd.read_csv('NIFTY50_all.csv', sep=',')
print(data.head())

# As we can see, no column names. Hence inserting column names

data = pd.read_csv('NIFTY50_all.csv', sep=',', header=None)
column_names = ['Date', 'Symbol', 'Series', 'Prev_Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume',
                'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
data.columns = column_names  # setting header names
print(data.head())

# Before we actually start training our model, we have to check the validations of the data. \
# As we can see in the above data there are certain values in certain columns that needs to \
# be modified in order to perform mathematical computations. Hence we replace those values with \
# the median of that feature column(In layman's term we will replace '?', 'NaN' with the \
# value or quantity lying at the midpoint of a frequency distribution of observed values or\
# quantities, such that there is an equal probability of falling above or below it.)

data['VWAP'].replace('?', np.NaN, inplace=True)
data['VWAP'].replace(np.NaN, data['VWAP'].median(), inplace=True)
data['High'].replace('?', np.NaN, inplace=True)
data['High'].replace(np.NaN, data['High'].median(), inplace=True)
data['Deliverable Volume'].replace(np.NaN, data['Deliverable Volume'].median(), inplace=True)
data['%Deliverble'].replace(np.NaN, data['%Deliverble'].median(), inplace=True)
data['Trades'].replace(np.NaN, data['Trades'].median(), inplace=True)
print(data.head())

print(data.info())

#  As we can see, 'High' & 'VWAP' features are not mathematically computable; so we convert those in float datatype

data['High'] = data['High'].astype('float64')
data['VWAP'] = data['VWAP'].astype('float64')
print(data.info())

# Data Visualization

print(data.corr())

style.use('ggplot')

print(data.describe())
X = np.array(data.drop(['Date', 'Symbol', 'Series', 'Close'], axis=1))
y = np.array(data['Close'])


# splitting training and testing data

def train_test_split(features, target, test_size=0.2):
    total_num_of_rows = features.shape[0]
    no_of_test_rows = int(total_num_of_rows * test_size)
    rand_row_num = np.random.randint(0, total_num_of_rows, no_of_test_rows)

    features_test = np.array([features[i] for i in rand_row_num])
    features_train = np.delete(features, rand_row_num, axis=0)

    target_test = np.array([target[i] for i in rand_row_num])
    target_train = np.delete(target, rand_row_num, axis=0)

    return features_train, features_test, target_train, target_test


# Now that we have separate training and testing datasets, we will keep the testing dataset aside and only use \
# it for testing purposes. All the training and optimization will be performed on the training dataset. \
# Also our data is widely distributed over a large region within a range(1e-4, 3e+7) and hence \
# we will normalize the data to get a mean of zero and standard \
# deviation of 1.


def StandardScaler(arr):
    arr1 = arr
    try:
        n = arr1.shape[1]
        for i in range(n):
            temp_arr = arr1[:, i]
            mean = temp_arr.mean()
            std = temp_arr.std()
            arr1[:, i] = (arr1[:, i] - mean) / std

    except IndexError:
        mean = arr.mean()
        std = arr.std()
        arr1 = (arr1 - mean) / std

    return arr1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_test_copy = y_test
feature_std = []
feature_mean = []
for i in range(X_test.shape[1]):
    feature_mean.append(X_test[:, i].mean())
    feature_std.append(X_test[:, i].std())

X_train = StandardScaler(X_train)
y_train = StandardScaler(y_train)
y_test = StandardScaler(y_test)
X_test = StandardScaler(X_test)

print('X_train (Mean):', X_train[:, 0].mean())
print('X_train (Standard_deviation):', X_train[:, 0].std())

print('X_test (Mean):', X_test[:, 0].mean())
print('X_test (Standard_deviation):', X_test[:, 0].std())

print('y_train (Mean):', y_train.mean())
print('y_train (Standard_deviation):', y_train.std())

print('y_test (Mean):', y_test.mean())
print('y_test (Standard_deviation):', y_test.std())


# Observe that each testing and training set has a mean of 0 and standard deviation of 1.


class LinearRegressionModel:
    def __init__(self, no_of_features, epochs, no_of_targets=1, learning_rate=0.1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.no_of_weights = no_of_features  # selecting number of weights
        self.weights = np.random.rand(no_of_features)
        self.bias = np.random.rand(no_of_targets)
        self.final_param = {}

    def fit(self, X_train, X_test, y_train, y_test):

        def LinearRegression(X_train):
            """calculates the y_hat predicted values on given weights and biases

            args-
                X_train = training dataset of features

                self.weights = Matrix/array of weights associated with each dependant variable

                self.bias = bias of the model

            returns -
                prediction = arra/matrix of predictions

            """
            predictions = self.weights.dot(np.transpose(X_train)) + self.bias
            return predictions

        def MSE(predictions):
            """calclulates Mean Squared Error and indicates inaccuracy of the model


            args -
                predictions = array/matrix of predictions

                y_train = actual values of data(target)

            returns -
                MSE = Mean Squared Error for given feature values


            """
            MSE = np.sum((predictions - y_train) ** 2 / len(y_train))
            return MSE

        predictions = LinearRegression(X_train)
        MSE_VAL = MSE(predictions)
        print('MSE_VALUE at random is ', MSE_VAL)

        def gradient():
            """calculates gradient of for weights and biases

            """
            n = len(X_train)
            predictions = LinearRegression(X_train)
            loss = y_train - predictions

            grad_bias = np.array([-2 / n * np.sum(loss)])
            grad_weights = np.ones(self.no_of_weights)
            for i in range(self.no_of_weights):
                featurecol = X_train[:, i]
                grad_weights[i] = -2 / n * np.sum(loss * featurecol)

            return grad_weights, grad_bias

        def stachscalerModified():
            """
            Performs stochastic gradient descent optimization on the model to give ideal weights and bias

            args -
                self.epochs = Number of times process to be repeated
                self.learning_rate = Size of steps during optimization
                self.weights = Matrix/array of weights associated with each dependant variable
                self.bias = bias of the model

            returns -
                return_dict
                keys = [weights, bias, MSE new value, MSE_list]
                values = [Final optimized weight, final optimized bias, Final MSE value, MSE list in optimization process]

            """
            MSE_list = []

            for i in range(self.epochs):
                grad_weights, grad_bias = gradient()
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
                new_predictions = LinearRegression(X_train)
                MSE_new = MSE(new_predictions)
                MSE_list.append(MSE_new)

            return_dict = {'weights': self.weights, 'bias': self.bias[0], 'MSE new value': MSE_new,
                           'MSE_list': MSE_list}
            return return_dict

        self.final_param = stachscalerModified()
        print('Final value of MSE ', self.final_param['MSE new value'])

    def predict(self, featureset):
        """final predictions using optimized weights and bias

        args -
            featureset = feature values
            self.weights = Matrix/array of weights associated with each dependant variable
            self.bias = bias of the model

        returns -
            predicted values matrix/array

        """
        prediction = np.sum(self.final_param['weights'] * featureset) + self.final_param['bias']
        return prediction

    def r2_score(self, y_test):
        """calculates accuracy of the model by using r squared method

        args -
            y_test = actual target test values


        returns -
            acc = accuracy of the model in percent



        """
        predicted = []
        for i in range(len(y_test)):
            predictions = self.predict(X_test[i])
            predicted.append(predictions)

        predicted = np.array(predicted)
        return (1 - np.sum((y_test - predicted) ** 2) / np.sum((y_test - y_test.mean()) ** 2)) * 100


# In order to find best model with highest accuracy, training process is repeated several times and \
# the model with best accuracy is pickled dwn in the file 'Bestfitmodel.pickle' and the same is embedded \
# in the web-app developed using streamlit

# below is the model training process:

# for i in range(10):
# '''model = LinearRegressionModel(11, 1000, learning_rate=0.01)
# model.fit(X_train, X_test, y_train, y_test)

# acc = model.r2_score(y_test)

# print('accuracy of the model is:', acc)

# with open('Bestfitmodel.pickle', 'wb') as f:
# pickle.dump(model, f)'''

pickle_in = open('Bestfitmodel.pickle', 'rb')
model = pickle.load(pickle_in)
print(model.r2_score(y_test))

# Here our training part is completed and now we will move on to web-app development, where data-visualization is
# also done.


nav_choice = st.sidebar.radio('Navigation', ('Home', 'Data Analysis', 'Predict'), index=0)

if nav_choice == 'Home':
    st.image("Nifty-50.jpg", width=800)

    st.success('Here, for the purpose of prediction, this app uses Linear Regression algorithm − '
               'one of the classic supervised machine learning algorithms.')

    st.warning('For the purpose of prediction, only features given in the table '
                'below are used. Detailed description about the features is provided within the table.')

    st.markdown('<table>'
                '<tr>'
                '<th align=\'centre\'><b>Features</b></th>'
                '<th align=\'centre\'><b>Description</b></th>'
                '</tr>'
                '<tr>'
                '<td>Prev_Close</td> <td>Previous Closing Value of Nifty-50</td>'
                '</tr>'
                '<td>Open</td> <td>Opening Price For current month</td>'
                '<tr>'
                '<td>High</td> <td>Highest price for the month</td>'
                '</tr>'
                '<td>Low</td><td>Lowest Price for the month</td>'
                '<tr>'
                '<td>Last</td><td>Last month\'s closing price for Nifty-50</td>'
                '</tr>'
                '<td>VWAP</td><td>The volume weighted average price (VWAP) is a trading benchmark used by traders that gives '
                'the average price a security has traded at throughout the day, based on both volume and price. '
                'It is important because '
                'it provides traders with insight into both the trend and value of a security</td>'
                '<tr>'
                '<td>Volume</td><td>Volume is the number of shares of a security traded during a given period of time.</td>'
                '</tr>'
                '<td>Turnover</td><td>Turnover for a particular month</td>'
                '<tr>'
                '<td>Trades</td><td>number of trades during month\'s period</td>'
                '</tr>'
                '<td>Deliverable Volume</td><td>Deliverable quantity or Deliverable Volume is the quantity of'
                ' shares which actually move from one set of people to another set of people</td>'
                '<tr>'
                '<td>%Deliverble</td><td>Percentage of deliverable volume</td>'
                '</tr>'
                '</table><br>', unsafe_allow_html=True)

    st.markdown('<b><font color=\'red\'>Given below is the link of the data used for the purpose of training of the model'
                '</font></b>'
                '<br><a href=\'https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=WIPRO.csv\' '
                'target=\'_blank\'>NIFTY50_all</a>'
                , unsafe_allow_html=True)


elif nav_choice == 'Data Analysis':
    un = np.array(data['Symbol'])
    un = np.unique(un)
    un = np.append(un, 'All')
    ind = []
    company_select = st.selectbox('Company', un, index=0)
    if company_select == 'All':
        data1 = data
    else:
        for i in range(len(un)):
            if un[i] == company_select:
                for j in range(len(data['Symbol'])):
                    if un[i] == data['Symbol'][j]:
                        ind.append(j)

        data1 = data.filter(ind, axis=0)
        print(len(data1), len(data))

    plot_choice = st.selectbox('Plots', ('Prev_Close vs Closing', 'Opening value vs Closing',
                                         'Highest value vs Closing', 'Lowest value vs Closing',
                                         'Last month closing vs Closing', 'VWAP vs Closing',
                                         'Volume vs Closing', 'Turnover vs Closing', 'Trades vs Closing',
                                         'Deliverable Volume vs Closing', '%Deliverble vs Closing'), index=0)
    load = st.progress(0)
    style.use('ggplot')

    if plot_choice == 'Prev_Close vs Closing':
        plt.scatter(data1['Prev_Close'], data1['Close'], c='r')
        plt.xlabel('Prev_close')
        plt.ylabel('Close')
        plt.title('Prev_Close vs Closing')
    elif plot_choice == 'Opening value vs Closing':
        plt.scatter(data1['Open'], data1['Close'], c='b')
        plt.xlabel('Open')
        plt.ylabel('Close')
        plt.title('Opening value vs Closing')
    elif plot_choice == 'Highest value vs Closing':
        plt.scatter(data1['High'], data1['Close'], c='g')
        plt.xlabel('Highest value')
        plt.ylabel('Close')
        plt.title('Highest value vs Closing')
    elif plot_choice == 'Lowest value vs Closing':
        plt.scatter(data1['Low'], data1['Close'], c='c')
        plt.xlabel('Lowest value')
        plt.ylabel('Close')
        plt.title('Lowest value vs Closing')
    elif plot_choice == 'Last month closing vs Closing':
        plt.scatter(data1['Last'], data1['Close'], c='chocolate')
        plt.xlabel('Last cosing value')
        plt.ylabel('Close')
        plt.title('Last month closing vs Closing')
    elif plot_choice == 'VWAP vs Closing':
        plt.scatter(data1['VWAP'], data1['Close'], c='y')
        plt.xlabel('VWAP')
        plt.ylabel('Close')
        plt.title('VWAP vs Closing')
    elif plot_choice == 'Volume vs Closing':
        plt.scatter(data1['Volume'], data1['Close'], c='k')
        plt.xlabel('Volume')
        plt.ylabel('Close')
        plt.title('Volume vs Closing')
    elif plot_choice == 'Turnover vs Closing':
        plt.scatter(data1['Turnover'], data1['Close'], c='magenta')
        plt.xlabel('Turnover')
        plt.ylabel('Close')
        plt.title('Turnover vs Closing')
    elif plot_choice == 'Trades vs Closing':
        plt.scatter(data1['Trades'], data1['Close'], c='b')
        plt.title('Trades vs Closing')
        plt.xlabel('Trades')
        plt.ylabel('Close')
    elif plot_choice == 'Deliverable Volume vs Closing':
        plt.scatter(data1['Deliverable Volume'], data1['Close'], c='r')
        plt.title('Deliverable Volume vs Closing')
        plt.xlabel('Deliverable Volume')
        plt.ylabel('Close')
    elif plot_choice == '%Deliverble vs Closing':
        plt.scatter(data1['%Deliverble'], data1['Close'], c='g')
        plt.title('%Deliverble vs Closing')
        plt.xlabel('%Deliverble')
        plt.ylabel('Close')

    st.pyplot()

    for i in range(100):
        time.sleep(0.001)
        load.progress(i + 1)

    st.header('Data Statistics')
    bar = st.slider('Select data Head size', 5, len(data1), 5, 1, key='data_head_size')
    data_show = st.checkbox('Show Data')
    data_corr = st.checkbox('Data Correlation')
    if data_show:
        st.success(f'Showing data head of size {bar}')
        st.table(data1.head(bar))
    if data_corr:
        st.table(data1.corr())

elif nav_choice == 'Predict':
    st.markdown('Kindly hit ENTER after each entry')
    prev = st.number_input('Previous Month Closing')
    open1 = st.number_input('Opening Amount')
    highest = st.number_input('Highest Value of Month')

    low = st.number_input('Lowest Value of Month')
    last = st.number_input('Last Month Highest')
    VWAP = st.number_input('VWAP')

    vol = st.number_input('Volume')
    turnover = st.number_input('Turnover')
    trades = st.number_input('Trades')

    del_vol = st.number_input('Deliverable Volume')
    perc_del_vol = st.number_input('%Deliverble')

    submit = st.button('Predict')


    def featureset_scale(arr):
        for i in range(X_test.shape[1]):
            arr[i] = (arr[i] - feature_mean[i]) / feature_std[i]
        return arr


    if submit:
        featureset = [prev, open1, highest, low, last, VWAP, vol, turnover, trades, del_vol, perc_del_vol]
        featureset = np.array(featureset)

        featureset = featureset_scale(featureset)

        prediction = model.predict(featureset)
        prediction = prediction * y_test_copy.std() + y_test_copy.mean()
        load = st.progress(0)
        for i in range(100):
            time.sleep(0.00001)
            load.progress(i + 1)
        st.success(f'Nifty-50 closing value is {prediction}')
