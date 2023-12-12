#!/usr/bin/env python
# coding: utf-8

# # E-mail Spam Detection

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("C:\\Users\\divi\\Downloads\\mail_data.csv")


# In[4]:


df.head()


# In[8]:


df.shape


# In[9]:


# data cleaning
# EDA
# Text preprocessing
# Model building
# Improvment
# Website
# deploy


# # 1.Data Cleaning
# 

# In[10]:


df.info()


# In[11]:


df.sample(3)


# In[13]:


from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()


# In[14]:


df['Category']=encoder.fit_transform(df['Category'])


# In[15]:


df.head()


# In[16]:


#missing values

df.isnull().sum()


# In[17]:


#check the duplicate values

df.duplicated().sum()


# In[20]:


df=df.drop_duplicates(keep='first')


# In[21]:


df.duplicated().sum()


# In[22]:


df.shape


# # 2.EDA

# In[23]:


df.head()


# In[24]:


df['Category'].value_counts()


# In[26]:


import matplotlib.pyplot as plt

plt.pie(df['Category'].value_counts(),labels=['ham','spam'],autopct='%0.2f')


# In[27]:


#data is imbalanced


# In[28]:


import nltk


# In[30]:


nltk.download('punkt')


# In[39]:


df['num_characters']=df["Message"].apply(len)


# In[40]:


df['num_characters']


# In[41]:


df.head()


# In[43]:


df['num_words']=df['Message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[44]:


df.head()


# In[45]:


df['num_sentences']=df['Message'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[46]:


df.head()


# In[48]:


df[['num_characters','num_words','num_sentences']].describe()


# In[54]:


#ham
df[df['Category']==0][['num_characters','num_words','num_sentences']].describe()


# In[55]:


#spam
df[df['Category']==1][['num_characters','num_words','num_sentences']].describe()


# In[56]:


import seaborn as sns


# In[67]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['Category']==0]['num_characters'],color='green')
sns.histplot(df[df['Category']==1]['num_characters'],color='red')


# In[68]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['Category']==0]['num_words'],color='green')
sns.histplot(df[df['Category']==1]['num_words'],color='red')


# In[70]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['Category']==0]['num_sentences'],color='green')
sns.histplot(df[df['Category']==1]['num_sentences'],color='red')


# In[71]:


sns.pairplot(df,hue='Category')


# In[77]:


sns.heatmap(df.corr(),annot=True)


# In[194]:


df.drop(columns=['text'],inplace=True)


# In[195]:


df.head()


# In[196]:


df.rename(columns={'Message':'text'},inplace=True)


# In[197]:


df.head()


# # 3.data preprocessing
# 
# *lower case
# *tokenization
# *Removing special characters
# *Removing stop words and punction
# *stemming

# In[248]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
            
    text=y[:]
    y.clear()
    for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
                
            text=y[:]
            y.clear()
            
            for i in text:
                y.append(ps.stem(i))
            
    return " ".join(y)


# In[249]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[250]:


import string 
string.punctuation


# In[257]:


transform_text(" i want to present a gift for my friend birthday give any idea,'Todays Voda numbers ending 7548 are selected to receive a $350 award. If you have a match please call 08712300220 quoting claim code 4041 standard rates app'")


# In[258]:


df['text'][123]


# In[259]:


from nltk.stem import PorterStemmer

ps = PorterStemmer()
print(ps.stem('loving'))


# In[260]:


df['text'].apply(transform_text)


# In[262]:


df.head()


# In[269]:


from wordcloud import WordCloud

wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[270]:


import pandas as pd

# Assuming 'transform_text' is the column you're working with
df['transform_text'] = df['transform_text'].astype(str)

# Now you can use .str.cat without encountering the TypeError
spam_wc = wc.generate(df[df['Category'] == 1]['transform_text'].str.cat(sep=""))


# In[272]:


plt.figure(figsize=(12,6))
plt.imshow(spam_wc)


# In[274]:


ham_wc = wc.generate(df[df['Category'] == 0]['transform_text'].str.cat(sep=""))


# In[275]:


plt.figure(figsize=(12,6))
plt.imshow(ham_wc)


# In[276]:


df.head()


# In[288]:


spam_corpus=[]
for msg in df[df['Category'] == 1]['transform_text'].tolist():
    for words in msg.split():
        spam_corpus.append(words)


# In[289]:


len(spam_corpus)


# In[291]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')


# In[292]:


ham_corpus=[]
for msg in df[df['Category'] == 0]['transform_text'].tolist():
    for words in msg.split():
        ham_corpus.append(words)


# In[293]:


len(ham_corpus)


# In[294]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')


# In[296]:


df.head(3)


# # 4.model building

# In[323]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf = TfidfVectorizer()


# In[324]:


x=tfidf.fit_transform(df['transform_text']).toarray()


# In[325]:


x.shape


# In[326]:


y=df['Category'].values


# In[327]:


y


# In[330]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[331]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[332]:


gnb.fit(x_train,y_train)
y_pred1=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[333]:


mnb.fit(x_train,y_train)
y_pred2=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[334]:


bnb.fit(x_train,y_train)
y_pred3=bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[335]:


df.head()


# In[339]:


df.loc[df['Category']=='spam','Category',]=0
df.loc[df['text']=='ham','Category',]==1


# In[340]:


x=df['text']

y=df['Category']


# In[341]:


print(x)


# In[342]:


print(y)


# In[346]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)


# In[347]:


print(x.shape)
print(x_train.shape)
print(x_test.shape)


# In[348]:


print(y.shape)
print(y_train.shape)
print(y_test.shape)


# In[354]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)

x_train_feature=feature_extraction.fit_transform(x_train)
x_test_feature=feature_extraction.transform(x_test)

y_train=y_train.astype('str')
y_test=y_test.astype('str')


# In[355]:


print(x_train)


# In[357]:


print(x_train_feature)


# In[359]:


from sklearn.linear_model import LogisticRegression


# In[360]:


model=LogisticRegression()


# In[364]:


model.fit(x_train_feature,y_train)


# In[365]:


prediction_on_training_data=model.predict(x_train_feature)

accuracy_on_training_data=accuracy_score(y_train,prediction_on_training_data)


# In[367]:


print("acc on training data :",accuracy_on_training_data)


# In[369]:


prediction_on_test_data=model.predict(x_test_feature)

accuracy_on_test_data=accuracy_score(y_test,prediction_on_test_data)


# In[370]:


print("acc on training data :",accuracy_on_test_data)


# In[377]:


email=[' I am taking half day leave bec i am not well']

input_data_features=feature_extraction.transform(email)

prediction=model.predict(input_data_features)

print(prediction)


if(prediction[0]==1):
    print("ham mail")
else:
    print("spam  mail")
    


# In[ ]:




