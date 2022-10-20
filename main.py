#Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


data = pd.read_csv("Instagram.csv", encoding='latin1')
print(data.head())

#Check whether the data contains any null values & drop all null values
data.isnull().sum()
data = data.dropna()

#Look at insights of columns to understand data types
data.info()

#Analyse the reach of instagram post, look at the distribution of impressions
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()

#distributions of impressions recieved from hashtags
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()

#distribution of impressions recieved from the explore section
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()

#percentage of impressions from various sources on instagram
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels,
             title='Impressions on Instagram Post From Various Sources')
fig.show()

#Analysing content, creating a wordcloud of the caption column to look at most used words
text = "".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#create another wordcloud of hashtag column, most used hashtags
text = "".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Analyse relationships to find most important factors of instagram reach, between likes and number of impressions
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols",
                    title="Relationship Between Likes and Impressions")
plt.show()

#relationship bewtween the number of comments and impressions
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols",
                    title="Relationship Between Comments and Total Impressions")
plt.show()

#relationship between the number of shares and impressions
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols",
                    title="Relationship Between Shares and Total Impressions")
plt.show()

#relationship between the numbers of saves and impressions
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols",
                    title="Relationship Between Post Saves and Total Impressions")
plt.show()

#correlation of all columns with the impression column
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

#Conversion Rates, formula=(Follows/Profile Visits) * 100
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)
#This conversion rate is 31%

#relationship betweem the total profile visits and followers gained
figure = px.scatter(data_frame=data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols",
                    title="Relationship Between Profile Visits and Followers Gained")

plt.show()

#training instagram model to predict the reach of an instagram post, split data into test sets before training
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares',
                    'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)


