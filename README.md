# Python-week-hackathon-nifty-50-predictor-linear-regression-ML
Predicts Stock price data of the fifty stocks in NIFTY-50 index from NSE India.<br>
In this end-to-end Machine Learning project-tutorial, I have created and trained a model from scratch, using NumPy, that uses the Linear Regression algorithm to predict the Nifty-50 closing price, further, the model with the best accuracy is embedded in the web-app developed using streamlit module for the purpose of prediction.   <br>
<h2>Understanding the Problem Statement</h2>
For this project, I have used the popular <a href="https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=WIPRO.csv" target="_blank">Nifty50_all</a> dataset for training the model and making predictions.<br>
For the purpose of prediction, only features given in the table below are used. Detailed description about the features is provided within the table.<br><br>
<table>
<tr>
  <th><b>Features</b></th>
  <th><b>Description</b></th>
</tr>
  <tr>
    <td>Prev_Close</td> <td>Previous Closing Value of Nifty-50</td>
    </tr>
    <td>Open</td> <td>Opening Price For current month</td>
  </tr>
    <td>High</td> <td>Highest price for the month</td>
    </tr>
    <td>Low</td><td>Lowest Price for the month</td>
    </tr>
    <td>Last</td><td>Last month's closing price for Nifty-50</td>
    </tr>
    <td>VWAP</td><td>The volume weighted average price (VWAP) is a trading benchmark used by traders that gives the average price a security has traded at throughout the day, based on both volume and price. It is important because it provides traders with insight into both the trend and value of a security</td>
    </tr>
    <td>Volume</td><td>Volume is the number of shares of a security traded during a given period of time.</td>
    </tr>
    <td>Turnover</td><td>Turnover for a particular month</td>
    </tr>
    <td>Trades</td><td>number of trades during month's period</td>
    </tr>
    <td>Deliverable Volume</td><td>Deliverable quantity or Deliverable Volume is the quantity of shares which actually move from one set of people to another set of people</td>
    </tr>
    <td>%Deliverble</td><td>Percentage of deliverable volume</td>
    </tr>
</table>

<h2>Key Project Takeaways</h2>
This project provided hands-on experience in real-time data handling and on the following Machine Learning Techniques:<br><br>
  <ul>Data wrangling for preprocessing and cleaning the training and testing data</ul>
  <ul>Normalizing the data</ul>
  <ul>Building an efficient Regression model from scratch using NumPy</ul>
  <ul>Mathematics behind SGD optimization</ul>
  <ul>Web-app development using streamlit module from python</ul>
