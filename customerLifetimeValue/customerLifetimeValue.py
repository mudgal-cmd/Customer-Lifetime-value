import pandas as pd
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_probability_alive_matrix
# importing 'lifetimes' library that's primarily built to aid users who are working with customer lifetime value
# calculations.
# It consists of various models (including the one being used in this project) and utility methods.
# Similar to scikit-learn and lifelines
import lifetimes as life

# NOTE:- Since the dataset contains approx 600,000 records, it may take up to 2 minutes to fetch the data.
data = pd.read_excel("/content/drive/MyDrive/OnlineRetail.xlsx")

# data.shape stores the dimension of the data as a tuple.
print(f"The data contains {data.shape[0]} rows and {data.shape[1]} columns")

data.head(10)

# For this project, only CustomerID, Invoice Date, Quantity, Invoice No, and Total Sales (Quantity*unit price)
# are needed
# hence, the remaining columns would be removed from the data.

# keeping only the required set of data
relevant_data = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]

# clv stores the data/columns relevant to our project
clv = data[relevant_data]
# clv["Total Sales"] = data_clv["Quantity"].multiply(data_clv["UnitPrice"])
clv["Total Sales"] = clv["Quantity"]*clv["UnitPrice"]
clv.head(10)

# descriptive statistics of the dataset
clv.describe()

# There are some negative values in 'quantity', 'unit price' and 'total sales' column,
# which could be due to the returned orders.
# Also, some discounts offered to customers could also be one of the contributing factors behind the negative values.
# We're concerned about only the positive values, hence removing the negative values from data_clv.
clv = clv[clv["Total Sales"] > 0]
clv.describe()

# Over 132,220 or ~25% of the records do not have a customer ID.
# We do not need these records, hence they will be removed from the data.

# removing null values from the dataframe.
clv = clv[pd.notnull(data["CustomerID"])]

pd.DataFrame(zip(clv.isnull().sum(), clv.isnull().sum()/len(clv)*100), columns=["Null Count", "Percentage"])

clv.head(10)

# printing a summary of the cleaned data

# most recent transaction date without time and timezone using 'datetime.date' object
recent_transaction_date = clv["InvoiceDate"].dt.date.max()
first_transaction_date = clv["InvoiceDate"].dt.date.min()

total_sale = clv["Total Sales"].sum()
# count only the unique customers, excluding any fields having Nan
total_unique_customers = clv["CustomerID"].nunique(dropna=True)

total_quantity_sold = clv["Quantity"].sum()
print(f"The data is from {first_transaction_date} to {recent_transaction_date}")
# print(first_transaction_date)
print("Total number of customers are: ", total_unique_customers)
print("Total revenue generated is : ", total_sale)
print("Total quantities sold is : ", total_quantity_sold)


# creating a RFM summary from the transactional data.
# RFM stands for - Recency, Frequency and Monetary value.

clv_summary = life.utils.summary_data_from_transaction_data(clv, "CustomerID", "InvoiceDate", "Total Sales")
clv_summary = clv_summary.reset_index()

clv_summary.head(10)
# 'T' represents the age of the customer in whatever time units chosen(here months)
# Here 0 in frequency and recency means that these users are one-time buyers.


# Creating a distribution of frequency to understand the customer frequency.

clv_summary["frequency"].plot.hist(stacked=True, bins=50)
print(clv_summary["frequency"].describe())

# Next we'd like to see how many such one-time buyers are there.
one_time_buyers = round(sum(clv_summary["frequency"] == 0)/float(len(clv_summary))*100, 2)
print(f"The percentage of one time buyers is: {one_time_buyers}%")


# fitting the BG/NBD model to the summary data
# BG/NBD model is available as 'BetaGeoFitter' class in lifetimes package
betageofitter = life.BetaGeoFitter(penalizer_coef=0.0)
betageofitter.fit(clv_summary["frequency"], clv_summary["recency"], clv_summary["T"])

print(betageofitter.summary)

# compute the probability of customer being alive
clv_summary["cust_probability_alive"] = betageofitter.conditional_probability_alive(clv_summary["frequency"],clv_summary["recency"], clv_summary["T"])
clv_summary.head(20)

# This graph depicts the relationship between frequency and recency

graph = plt.figure(figsize=(9, 7))
plot_probability_alive_matrix(betageofitter)

t = 30
# Predicting the number of transactions a customer will do using the
# conditional_expected_number_of_purchases_up_to_time() method
clv_summary['predicted_txn_count'] = round(betageofitter.conditional_expected_number_of_purchases_up_to_time
                                           (t, clv_summary['frequency'], clv_summary['recency'], clv_summary['T']),2)
clv_summary.sort_values(by='predicted_txn_count', ascending=False).head(10).reset_index()

# Now we'd like to see how frequency and monetary value are related if at all they are.
frequent_buyers = clv_summary[clv_summary["frequency"] > 0]
print(frequent_buyers.head())
frequent_buyers[["frequency", "monetary_value"]].corr()

# The correlation looks weak which implies that the model can be fit to the data.
gamma_gamma_model = life.GammaGammaFitter(penalizer_coef=0.001)
gamma_gamma_model.fit(frequent_buyers["frequency"], frequent_buyers["monetary_value"])

print(gamma_gamma_model.summary)

# Now, to calculate the profit made by the company per customer per txn.

clv_summary = clv_summary[clv_summary["monetary_value"]>0]
clv_summary["predicted_sales"] = gamma_gamma_model.conditional_expected_average_profit(clv_summary["frequency"]
                                                                                       , clv_summary["monetary_value"])
clv_summary.head(10)

# Now, we'd like to check if our predicted average value is close to the actual average sales.
print(f'Predicted mean sales of the company is: {clv_summary["predicted_sales"].mean():.4f}')
print(f'Actual mean sales of the company is: {clv_summary["monetary_value"].mean():.4f} ')

# The predicted mean value of average sale is significantly close to the actual value.

# Predicting the value of CLV for every customer for the next 2 months
clv_summary["pred_customer_lifetime_value"] = gamma_gamma_model.customer_lifetime_value(betageofitter,
                                                                                        clv_summary["frequency"],
                                                                                        clv_summary["recency"],
                                                                                        clv_summary["T"],
                                                                                        clv_summary["monetary_value"],
                                                                                        # lifetime (here in months)
                                                                                        time=2,
                                                                                        # since we have a daily data, hence, 'D'
                                                                                        freq="D",
                                                                                        # in case of monthly data, use 'M'
                                                                                        discount_rate = 0.0105)
# The final summary illustrates the customer lifetime value for each customer.
clv_summary.head(20)


