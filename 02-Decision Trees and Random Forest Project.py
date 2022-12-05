#!/usr/bin/env python
# coding: utf-8

# # Random Forest Project 
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns


# ## Get the Data
# 
# ** Use pandas to read loan_data.csv as a dataframe called loans.**

# In[3]:


loans= pd.read_csv(r'C:\CSV_files\loan_data.csv')


# ** Check out the info(), head(), and describe() methods on loans.**

# In[4]:


loans.head()


# In[5]:


loans.info()


# In[6]:


loans.describe()


# # Exploratory Data Analysis
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# ** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# *Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*

# In[7]:


loans.hist(column='fico')


# In[8]:


loans.hist(column='fico', by='credit.policy', bins=25, grid=False, figsize=(8,10), layout=(3,1), sharex=True,color='#86bf91', zorder=2, rwidth=0.9)


# ** Create a similar figure, except this time select by the not.fully.paid column.**

# In[9]:


loans.hist(column='not.fully.paid', by='credit.policy', bins=25, grid=False, figsize=(8,10), layout=(3,1), sharex=True,color='#86bf91', zorder=2, rwidth=0.9)


# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

# In[10]:


sns.countplot(x ='purpose', data = loans, hue='not.fully.paid')


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**

# In[11]:


sns.jointplot(x = "fico", y = "int.rate", data = loans)


# ** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**

# In[12]:


#https://www.geeksforgeeks.org/python-seaborn-lmplot-method/
sns.lmplot(x='not.fully.paid', y='credit.policy',data=loans)


# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

# In[13]:


loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**

# In[14]:


cat_feats = loans['purpose'].to_list()


# In[15]:


print(cat_feats)


# In[16]:


type(cat_feats)


# **Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**

# In[17]:


pd.get_dummies(cat_feats)


# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**

# In[28]:


x, y = np.arange(10).reshape((5, 2)), range(5)


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# ** Import DecisionTreeClassifier**

# In[39]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:





# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

# In[44]:


dtree= DecisionTreeClassifier()


# In[45]:


dtree = dtree.fit(x_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**

# In[46]:


y_pred = dtree.predict(x_test)


# In[47]:


print(y_pred)


# In[48]:


cm = confusion_matrix(y_test, y_pred)


# In[49]:


sns.heatmap(cm, annot=True, cmap='Blues')


# In[50]:


cm


# In[53]:


y_pred_train = dtree.predict(x_train)
print(classification_report(y_train, y_pred_train))


# In[55]:


print(classification_report(y_test, y_pred))


# ## Training the Random Forest model
# 
# Now its time to train our model!
# 
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**

# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[80]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100) 


# In[81]:


rfc= rfc.fit(x_train, y_train)


# In[82]:


y_pred = rfc.predict(x_test)


# In[83]:


print(y_pred)


# ## Predictions and Evaluation
# 
# Let's predict off the y_test values and evaluate our model.
# 
# ** Predict the class of not.fully.paid for the X_test data.**

# In[84]:


y_test = rfc.predict(x_test)


# In[85]:


loans[['not.fully.paid']] = y_test


# In[87]:





# **Now create a classification report from the results. Do you get anything strange or some sort of warning?**

# In[86]:


print(classification_report(y_train, y_pred_train))


# In[ ]:





# **Show the Confusion Matrix for the predictions.**

# In[ ]:





# **What performed better the random forest or the decision tree?**

# In[ ]:




