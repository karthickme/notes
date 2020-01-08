from sklearn import preprocessing
import pandas as pd


#ChurnData.csv
#We will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it is less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company.
# This data set provides information to help you predict what behavior will help you to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.
# The dataset includes information about:
#     Customers who left within the last month – the column is called Churn
#     Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
#     Customer account information – how long they had been a customer, contract, payment method, paperless billing, monthly charges, and total charges
#     Demographic info about customers – gender, age range, and if they have partners and dependents


churn_df = pd.read_csv("ChurnData.csv")

#Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

#Normalization
X = preprocessing.StandardScaler().fit(X).transform(X)

print("Normalized DataSet:",X)