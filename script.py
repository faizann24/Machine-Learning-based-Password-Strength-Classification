import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def getTokens(inputString): #custom tokenizer. ours tokens are characters rather than full words
	tokens = []
	for i in inputString:
		tokens.append(i)
	return tokens
  
  
filepath = 'your_file_path_containing_passwords_and_labels' #path for password file
data = pd.read_csv(filepath,',',error_bad_lines=False)

data = pd.DataFrame(data)
passwords = np.array(data)

random.shuffle(passwords) #shuffling randomly for robustness
y = [d[1] for d in passwords] #labels
allpasswords= [d[0] for d in passwords] #actual passwords

vectorizer = TfidfVectorizer(tokenizer=getTokens) #vectorizing
X = vectorizer.fit_transform(allpasswords)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)  #splitting

lgs = LogisticRegression(penalty='l2',multi_class='ovr')  #our logistic regression classifier
lgs.fit(X_train, y_train) #training
print(lgs.score(X_test, y_test))  #testing

#more testing
X_predict = ['faizanahmad','faizanahmad123','faizanahmad##','ajd1348#28t**','ffffffffff','kuiqwasdi','uiquiuiiuiuiuiuiuiuiuiuiui','mynameisfaizan','mynameis123faizan#','faizan','123456','abcdef']
X_predict = vectorizer.transform(X_predict)
y_Predict = lgs.predict(X_predict)
print y_Predict
