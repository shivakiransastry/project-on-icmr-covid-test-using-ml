import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

icmr=pd.read_csv('ICMR_Testing_Data.csv')
icmr1=icmr.replace('[^\d.]','',regex=True).astype(int)

x=icmr1.iloc[:,[1,2,5,6]].values
y=icmr1.iloc[:,3].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=44)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)
a=int(input('enter date by ignoring symbols and space:'))
b=int(input('enter samples tested:'))
c=int(input('per daypositive:'))
d=int(input('perday tests:'))
p=reg.predict([[a,b,c,d]]).astype(int)
print('positive cases:',p)
    
pt.scatter(y_pred,y_test,label='y  prediction',color='black')
pt.plot(y_train,y_train,color='blue',label='y_trained')
pt.scatter(p,p,color='red',label='predicted ')
pt.title('performance of batsman')
pt.xlabel('predicted case',color='red',fontsize=14)
pt.ylabel('tested case',color='blue',fontsize=14)
pt.legend()
pt.grid(True)
pt.show()

print('efficience of the algorithm respect to dataset :',reg.score(x_test,y_test))

