# Import data music.csv for this project.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import tree
# To check accuracy
from sklearn.metrics import accuracy_score

# Clean the data - removing duplicates, null values
# separate te data set in 2 parts which we refer as 1st 2 columns as input set & last column as output set (contains our predictions)

# Now we can give a new set of values to predict the output | Ex: 21 Male

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns = ['genre'])
# print(pd.pivot_table(music_data,index = ["gender"], values = ["age"]))
# X
# Now create the output set
y = music_data['genre']
# Later part for 20% testing | Returns a tuple --> 4 variable so we can unpack it.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# X_train, X_test are input set for traning and testing and other are otput set for the same.
# y
# Learning and Predictions using an algorithm
# Using Decision tree Algorithm

model = DecisionTreeClassifier()
# model.fit(X,y)         # 100% data in training only which is not good
model.fit(X_train,y_train)
# music_data



# Model Creation
joblib.dump(model,'music-recommender.joblib')

# Model loading
joblib.load('music-recommender.joblib')

# Creating Visualization '.dot' file:
# tree.export_graphviz(model, out_file='music-recommender.dot',
#                     feature_names=['age','gender'],
#                     class_names=sorted(y.unique()),
#                     label='all',
#                     rounded=True,
#                     filled=True)


# with open("music-recommender.dot") as f:
#     dot_graph = f.read()

# DOT file creation
# graphviz.Source(dot_graph)

# To check .dot file online --> copy paste the code directly on the below link
# http://www.webgraphviz.com/
# https://dreampuf.github.io/GraphvizOnline/#digraph%20Tree%20%7B%0D%0Anode%20%5Bshape%3Dbox%2C%20style%3D%22filled%2C%20rounded%22%2C%20color%3D%22black%22%2C%20fontname%3Dhelvetica%5D%20%3B%0D%0Aedge%20%5Bfontname%3Dhelvetica%5D%20%3B%0D%0A0%20%5Blabel%3D%22age%20%3C%3D%2030.5%5Cngini%20%3D%200.778%5Cnsamples%20%3D%2018%5Cnvalue%20%3D%20%5B3%2C%206%2C%203%2C%203%2C%203%5D%5Cnclass%20%3D%20Classical%22%2C%20fillcolor%3D%22%23e5fad7%22%5D%20%3B%0D%0A1%20%5Blabel%3D%22age%20%3C%3D%2025.5%5Cngini%20%3D%200.75%5Cnsamples%20%3D%2012%5Cnvalue%20%3D%20%5B3%2C%200%2C%203%2C%203%2C%203%5D%5Cnclass%20%3D%20Acoustic%22%2C%20fillcolor%3D%22%23ffffff%22%5D%20%3B%0D%0A0%20-%3E%201%20%5Blabeldistance%3D2.5%2C%20labelangle%3D45%2C%20headlabel%3D%22True%22%5D%20%3B%0D%0A2%20%5Blabel%3D%22gender%20%3C%3D%200.5%5Cngini%20%3D%200.5%5Cnsamples%20%3D%206%5Cnvalue%20%3D%20%5B0%2C%200%2C%203%2C%203%2C%200%5D%5Cnclass%20%3D%20Dance%22%2C%20fillcolor%3D%22%23ffffff%22%5D%20%3B%0D%0A1%20-%3E%202%20%3B%0D%0A3%20%5Blabel%3D%22gini%20%3D%200.0%5Cnsamples%20%3D%203%5Cnvalue%20%3D%20%5B0%2C%200%2C%203%2C%200%2C%200%5D%5Cnclass%20%3D%20Dance%22%2C%20fillcolor%3D%22%2339e5c5%22%5D%20%3B%0D%0A2%20-%3E%203%20%3B%0D%0A4%20%5Blabel%3D%22gini%20%3D%200.0%5Cnsamples%20%3D%203%5Cnvalue%20%3D%20%5B0%2C%200%2C%200%2C%203%2C%200%5D%5Cnclass%20%3D%20HipHop%22%2C%20fillcolor%3D%22%233c39e5%22%5D%20%3B%0D%0A2%20-%3E%204%20%3B%0D%0A5%20%5Blabel%3D%22gender%20%3C%3D%200.5%5Cngini%20%3D%200.5%5Cnsamples%20%3D%206%5Cnvalue%20%3D%20%5B3%2C%200%2C%200%2C%200%2C%203%5D%5Cnclass%20%3D%20Acoustic%22%2C%20fillcolor%3D%22%23ffffff%22%5D%20%3B%0D%0A1%20-%3E%205%20%3B%0D%0A6%20%5Blabel%3D%22gini%20%3D%200.0%5Cnsamples%20%3D%203%5Cnvalue%20%3D%20%5B3%2C%200%2C%200%2C%200%2C%200%5D%5Cnclass%20%3D%20Acoustic%22%2C%20fillcolor%3D%22%23e58139%22%5D%20%3B%0D%0A5%20-%3E%206%20%3B%0D%0A7%20%5Blabel%3D%22gini%20%3D%200.0%5Cnsamples%20%3D%203%5Cnvalue%20%3D%20%5B0%2C%200%2C%200%2C%200%2C%203%5D%5Cnclass%20%3D%20Jazz%22%2C%20fillcolor%3D%22%23e539c0%22%5D%20%3B%0D%0A5%20-%3E%207%20%3B%0D%0A8%20%5Blabel%3D%22gini%20%3D%200.0%5Cnsamples%20%3D%206%5Cnvalue%20%3D%20%5B0%2C%206%2C%200%2C%200%2C%200%5D%5Cnclass%20%3D%20Classical%22%2C%20fillcolor%3D%22%237be539%22%5D%20%3B%0D%0A0%20-%3E%208%20%5Blabeldistance%3D2.5%2C%20labelangle%3D-45%2C%20headlabel%3D%22False%22%5D%20%3B%0D%0A%7D





# Ask model 21 M or 22 F
# predictions = model.predict([[21,1],[22,0]])
predictions = model.predict(X_test)
predictions

# Measure the model's accuracy
# Split the data set in 2 sets - training and testing
# General thumb rule is we keep 70-80% of data for training and other 20-30% for testing


score = accuracy_score(y_test,predictions)
print(f"Accuracy Score is {score * 100} %")



