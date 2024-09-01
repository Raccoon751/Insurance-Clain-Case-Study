#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as date
import scipy.stats as stats


# In[2]:


claims=pd.read_csv(r"C:\Users\Shiv_Shakti\OneDrive\Desktop\claims.csv")
claims


# In[3]:


claims.tail(3)


# In[4]:


cust_demographics=pd.read_csv(r"C:\Users\Shiv_Shakti0\OneDrive\Desktop\cust_demographics.csv")
cust_demographics


# In[5]:


cust_demographics.head(3)


# # 1. Import claims_data.csv and cust_data.csv which is provided to you and combine the two datasets appropriately to create a 360-degree view of the data. Use the same for the subsequent questions.

# In[6]:


cust_360=pd.merge(left=claims,right=cust_demographics,how="inner",left_on="customer_id",right_on="CUST_ID")


# In[7]:


df=cust_360
df # giving the cust_360 to df for convinience


# In[8]:


df.isnull().sum()


# # 2. Perform a data audit for the datatypes and find out if there are any mismatch within the current datatypes of the columns and their business significance.

# In[9]:


df.info()

here we need to change the data type of date columns 
# In[10]:


df.claim_date=pd.to_datetime(df.claim_date,format="%m/%d/%Y")


# In[11]:


df.info()


# In[12]:


df. DateOfBirth =pd.to_datetime(df.DateOfBirth ,format="%d-%b-%y")


# In[13]:


df.info()


# # 3. Convert the column claim_amount to numeric. Use the appropriate modules/attributes to remove the $ sign.
# 

# In[14]:


df['claim_amount'] = df['claim_amount'].str.replace('$', '')


# In[15]:


df.info()


# In[16]:


df['claim_amount'] = pd.to_numeric(df['claim_amount'])
# changing data type also


# In[17]:


df.info()


# # 4. Of all the injury claims, some of them have gone unreported with the police. Create an alert flag (1,0) for all such claims.

# In[18]:


df['unreported_claim']=np.where(df. police_report =='Unknown',1,0)


# In[19]:


df.head(3)


# # 5. One customer can claim for insurance more than once and in each claim,multiple categories of claims can be involved. However, customer ID should remain unique. Retain the most recent observation and delete any duplicated records inthe data based on the customer ID column.

# In[20]:


df=df.drop_duplicates(subset="customer_id",keep="last")


# In[21]:


df


# # 6. Check for missing values and impute the missing values with an appropriate value. (mean for continuous and mode for categorical)

# In[22]:


df.shape


# In[23]:


df_continuous = df.select_dtypes(include=['float64', 'int64'])
df_categorical = df.select_dtypes('object')


# In[24]:


df_continuous


# In[25]:


df_categorical


# In[26]:


df.isnull().sum()


# ####  here we have missing value in claim_amount and in total_policy_claims . ther eis no missing value in catagorical data

# In[27]:


df.info()


# In[28]:


df['claim_amount'].mean() 


# In[29]:


df["claim_amount"] = df['claim_amount'].fillna(df['claim_amount'].mean())


# In[30]:


df.isnull().sum()


# In[31]:


#df['total_policy_claims'].sum()
df['total_policy_claims'].mean()


# In[32]:


#df["total_policy_claims"]= df['total_policy_claims'].replace(np.NaN,df['total_policy_claims'].mean())


# In[33]:


df['total_policy_claims'].mean()


# In[34]:


df["total_policy_claims"]= df['total_policy_claims'].fillna(df['total_policy_claims'].mean())


# In[35]:


df['total_policy_claims'].mean()


# # 7. Calculate the age of customers in years. Based on the age, categorize thecustomers according to the below criteriaChildren < 18 ,Youth 18-30       Adult 30-60  , Senior > 60

# In[36]:


df.head(2)


# In[37]:


df["Age"] = (pd.DatetimeIndex(df.claim_date).year - pd.DatetimeIndex(df.DateOfBirth).year)
df.loc[(df.Age < 18) & (df.Age >0),'Age_Group'] = 'Children'
df.loc[(df.Age >=18) & (df.Age <30),'Age_Group'] = 'Youth'
df.loc[(df.Age >=30) & (df.Age <60),'Age_Group'] = 'Adult'
df.loc[(df.Age >=60),'Age_Group'] = 'Senior'
df


# # 8. What is the average amount claimed by the customers from various segments?
# 

# In[38]:


avg_amount=df.groupby(["Segment"])["claim_amount"].mean()
avg_amount


# # 9. What is the total claim amount based on incident cause for all the claims that have been done at least 20 days prior to 1st of October, 2018.
# 

# In[39]:


df.claim_date.max()


# In[40]:


round(df.loc[df['claim_date']<"2018-9-10"].groupby(["incident_cause"])["claim_amount"].sum(),2)#.add_prefix("total_") it will add total to all index


# # 10. How many adults from TX, DE and AK claimed insurance for driver related issues and causes? 

# In[41]:


Adults_claims_count=df.loc[df['State'].isin(['TX','DE','AK'])&df['incident_cause'].str.lower().str.contains("driver")].groupby(['State'])["claim_amount"].count()


# In[42]:


Adults_claims_count


# ### 11. Draw a pie chart between the aggregated value of claim amount based on gender and segment. Represent the claim amount as a percentage onthe pie chart.

# In[43]:


pie_claim=round(df.groupby(['gender','Segment'])["claim_amount"].sum(),2).reset_index()


# In[44]:


x=pd.DataFrame(pie_claim)


# In[45]:


x.T.reset_index()


# In[46]:


xx=pie_claim.pivot(index="Segment", columns= "gender", values= "claim_amount")


# In[47]:


xx.T


# In[48]:


xx.T.plot(kind="pie", subplots= True, legend= True,figsize=(20,10))
plt.show()


# In[49]:


get_ipython().run_line_magic('pinfo', 'plt.plot')


# # 12. Among males and females, which gender had claimed the most for any type of driver related issues? E.g. This metric can be compared using a bar chart
# 

# In[50]:


gender_count=df.loc[df['incident_cause'].str.lower().str.contains("driver")].groupby(['gender'])[["gender"]].count().add_prefix("countof_").reset_index()


# In[51]:


gender_count


# In[52]:


sns.barplot(x = "gender", y = "countof_gender", data = gender_count )
plt.show()


# # 13. Which age group had the maximum fraudulent policy claims? Visualize it on a bar chart.

# In[53]:


p2 = df.groupby("Age_Group")[["fraudulent"]].count().reset_index()
p2


# In[54]:


sns.barplot(x = 'Age_Group', y = 'fraudulent', data = p2)
plt.legend(['Adult','Childen','Youth'])
plt.title('Fraudulent cases by different age groups')
plt.show()


# # 14. Visualize the monthly trend of the total amount that has been claimed by the customers. Ensure that on the “month” axis, the month is in a chronological order not alphabetical order. 

# In[55]:


df


# In[56]:


df['month']=df['claim_date'].dt.strftime("%B")


# In[57]:


df.head(3)


# In[58]:


monthly_claim=round(df.groupby(["month"])[["claim_amount"]].sum().add_prefix("monthly_").reset_index().sort_values("monthly_claim_amount"),2)
monthly_claim


# In[59]:


#monthly_claim=monthly_claim.sort_values("monthly_claim_amount")
monthly_claim


# In[60]:


#gender_count.plot(kind="bar",x="gender",y="countof_gender")
monthly_claim.plot(kind="bar",x="month",y="monthly_claim_amount")
plt.title("Monthly claims analysis")


# # 15. What is the average claim amount for gender and age categories and suitably represent the above using a facetted bar chart, one facet that represents fraudulent claims and the other for non-fraudulent claims.

# In[61]:


f = df[(df.fraudulent=="Yes")].groupby(["gender","Age_Group"])[["claim_amount"]].mean().add_prefix("Fraud_")
nf = df[(df.fraudulent=="No")].groupby(["gender","Age_Group"])[["claim_amount"]].mean().add_prefix("Non_Fraud_")
f_nf=round(pd.merge(f,nf,on=["gender","Age_Group"]),2)
f_nf.plot(kind="bar", subplots= True, legend= True)
plt.show()


# # Based on the conclusions from exploratory analysis as well as suitable statistical tests, answer the below questions. Please include a detailed write-up on the parameters taken into consideration, the Hypothesistesting steps, conclusion from the p-values and the business implications of the statements. 

# ## 16. Is there any similarity in the amount claimed by males and females?
# 

# In[62]:


df.groupby("gender")[["claim_amount"]].sum()


# In[63]:


claim_male = df['claim_amount'].loc[df['gender']=="Male"]
claim_female = df['claim_amount'].loc[df['gender']=="Female"]


# # Two sample t-test:
# # To conduct a valid test: (Assumptions for two sample t-test)
# # 
# # * Data values must be independent. Measurements for one observation do not affect measurements for any other observation.
# # * Data in each group must be obtained via a random sample from the population.
# # * Data in each group are normally distributed.
# # * Data values are continuous.
# # * The variances for the two independent groups are equal.
# 

# In[65]:


eq_var = stats.ttest_ind(a= claim_male,
                b= claim_female,
                equal_var=True)    # equal variance
eq_var.statistic


# In[66]:


uneq_var = stats.ttest_ind(a= claim_male,
                b= claim_female,
                equal_var=False)    # UnEqual variance
uneq_var.statistic


# In[67]:


import scipy.stats as st
female = df['claim_amount'].loc[df['gender']=="Female"]
male = df['claim_amount'].loc[df['gender']=="Male"]
st.ttest_ind(female, male, equal_var=False
            )


# ## Hypothesis testing:
# - H0 = There is no similiarity in the aount claimed by males and females
# 
# - H1 = There is a similiarity in the amount claimed by males and females
# 
# - significance_level = 0.05
# 
# - test = ttest
# 

# ### conclusion
# - As the p-value is less than 0.05, we reject the null hypothesis. 
# - So there is a similiarity in the amount claimed by males and females

# ## 17] Is there any relationship between age category and segment?

# ### Hypothesis testing:
# - H0 = There is no relationship between age group and segment
# - h1 = There is a relationship between age group and segment
# - significance_level = 0.05
# - test = chi square test
# 

# In[72]:


c = pd.crosstab(df.Age_Group, df.Segment, margins = True)
st.chi2_contingency(observed= c)


# ## conclusion
# - As the p-value is greater than 0.05, we fail to reject the null hypothesis.
# - So there is no relationship between age_group and segments

# ## 18. The current year has shown a significant rise in claim amounts as compared to 2016-17 fiscal average which was $10,000.

# In[73]:


df['year'] = pd.DatetimeIndex(df.claim_date).year
CY = df.loc[df.year == 2018]["claim_amount"]
PY = df.loc[df.year == 2017]["claim_amount"]
CY.corr( PY)


# # conclusion
# ### There is no correlation 

# ## 19. Is there any difference between age groups and insurance claims?

# ## Hypothesis testing:
# - H0 = There is no difference between age groups and insurance claims
# - h1 = There is a difference between age group and insurance claims
# - test = anova

# In[76]:


a1 = df['total_policy_claims'].loc[df['Age_Group']=="Youth"]
a2 = df['total_policy_claims'].loc[df['Age_Group']=="Adult"]
st.f_oneway(a1,a2)


# ## conclusion
# - As the p-value is greater than 0.05, we fail to reject the null hypothesis.
# - So there is no difference between age groups and insurance claims

# ## 20] Is there any relationship between total number of policy claims and the claimed amount?

# In[77]:


df.total_policy_claims.corr(other= df.claim_amount)


# ##### negatively correlated between total number of policy claims and the claimed amount.
# - Thus there is no significantrelationship between the variables .

# In[ ]:




