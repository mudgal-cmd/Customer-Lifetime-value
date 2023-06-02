# Customer-Lifetime-value
Predicting Customer Lifetime value for a business using Python 


STATISTICAL MODEL USED:

For this project, the analysis would be carried out using the Beta Geometric/Negative Binomial Distribution (BG/NBD) Model.

The Recency, Frequency and Monetary value (RFM) summary from the transactional data contains the below paramters:

1) The term 'frequency' indicates the number of times a customer has made repeat purchases. This is calculated by subtracting one from the total number of purchases, excluding the first order. However, this understanding is slightly incorrect. In reality, it represents the count of time periods in which the customer made a purchase. For instance, if we consider days as units, it represents the count of days on which the customer made a purchase. 
2) The symbol 'T' represents the customer's age measured in chosen time units (e.g., monthly in our dataset). It signifies the duration between the customer's first purchase and the end of the analyzed period. On the other hand, 'recency' represents the customer's age at the time of their most recent purchase. It is calculated as the duration between the customer's first purchase and their latest purchase. In the case of a customer with only one purchase, the recency value would be 0. 
3) Lastly, 'monetary_value' denotes the average value of a customer's purchases. It is obtained by dividing the sum of all a customer's purchases by the total number of purchases.
