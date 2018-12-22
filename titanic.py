"""
타이타닉호의 침몰당시 승객명단데이터를 통해 생존자의 이름 성별 나이 티켓요금 생사여부의 정보를 획득합니다
이를 분석하여 각각의 데이터들간의 연관성을 분석하여
생존에 영향을 미치는 요소를 찾아내는 것
데이터는 train.csv(훈련데이터)와 test.csv(목적데이터) 두개가 제공됩니다.

목적데이터는 훈련데이터에서 survived 즉 생존여부에 대한 정보가 빠져있습니다.
즉 훈련데이터에 있는 정보를 통해서 적합한 분석 model을 구성한 뒤 이를 목적데이터에 반영하여 생존여부를 추측하는 과정을 수행하고자 합니다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns

train = pd.read_csv("C:\\ezen_tensorflow\\seoul_cctv\\seoul_cctv\\train.csv")
test= pd.read_csv("C:\\ezen_tensorflow\\seoul_cctv\\seoul_cctv\\test.csv")
train.head()
test.head()

train.columns
"""
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
PassengerId - 승객번호
Survived - 0사망 1생존
Pclass - 승성권 클래스 1:1등석 2:2등석 3:3등석
Name - 승객이름
Sex - 승객성별
Age - 승객나이
SibSp - 동반한 형제, 자매, 배우자 수
Parch - 동반한 부모, 자식수
Ticket - 티켓의 고유넘버
Fare - 티켓의 요금
Cabin - 객실번호
Embarked - 승선한 항구명 C: 캠브릿지 Q:퀸즈타운 S:사우스햄프턴
"""

f, ax = plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data= train, ax = ax[1])
ax[1].set_title('Survived')
plt.show()

'''
탑승객의 60 % 이상이 사망했음( 0사망, 1생존)
'''

f, ax = plt.subplots(1,2,figsize=(18,8))
train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1], shadow=True)
ax[0].set_title('남성생존자')
ax[1].set_title('여성생존자')
plt.show()


# 성별과 객실 클래스와의 관계시트생성하려고 할 때 crosstab 사용함
df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']
pd.crosstab(df_1, df_2, margins=True)

# 1등객실 여성으 ㅣ생존률은 91/94 = 97%
# 3등객실 여성의 생존률은 50%
# 1등객실 남성의 생존률은 37%
# 3등객실 남성의 생존률은 13%

# 배를 탄 항구와의 연관성 추출
f , ax = plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked', data=train, ax = ax[0,0])
ax[0,0].set_title('on_board_person')

sns.countplot('Embarked', hue='Sex', data=train, ax = ax[0,1])
ax[0,1].set_title('on_board_Sex')

sns.countplot('Embarked', hue='Survived', data=train, ax = ax[1,0])
ax[1,0].set_title('on_board_port vs Survived')

sns.countplot('Embarked', hue='Pclass',data=train, ax = ax[1,1])
ax[1,1].set_title('on_board_port vs Pclass')

'''
절반이상의 승객이 사우스햄프턴에서 배를 탓으며 여기에 탑승한 승객의 70% 가량이 남성이었습니다.
남성의 사망률이 여성보다 훨씬높았으므로 사우스햄프턴에서 탑승한 승객의 사망률이 높게 나왔습니다.
캠브릿지에서 탑승한 승객들은 1등 객실 승객의 비중 및 생존률이 높은 것으로보아 이동네느 부자동네임을 짐작하게 합니다.
'''