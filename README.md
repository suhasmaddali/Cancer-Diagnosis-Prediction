# Cancer-Diagnosis-Prediction

This project was developed with the help of AppliedAI team based in India. 

Based on the given set of features such as gene, mutation and text associated with a given data, the machine learning models would predict the type of cancer that a person would have under 9 classes. 


The machine learning models that were used in the process of prediction were K Nearest Neighbors, Linear Support Vector Machines (SVM), Logistic Regression, Stacking Classifer, Voting Classifier and Random Forests Classifiers. 


Given a set of input features such as gene, the type of mutation and the text that is associated with the gene and the mutation, the machine learning model could classify the changes of a person to suffer from cancer. In addition to this, it would also give an indication of the type of class of the cancer from 1 to 9. 



We have to be looking at different metrics in the problem before diagnosing the solution. We would be looking for some metrics such as log loss, accuracy and so on for different models and add those in tables so that we would be better able to decide the right kind of algorithm for the problem. 


![](Images/2020-12-22%20(1).png)

Here, we could see that the number of points that we have taken into consideration are 3321. We have just 2 features with the file that we have taken namely the ID and the Text associated with it. We see just 5 rows just to consider the head of the data frame. 

![](Images/2020-12-22%20(2).png)

We would be preprocessing the text and converting it into a mathematical vector so that those values could be given to the machine learning model. We first take into consideration the stop words which are very important in english. Later, we would be removing every special character as they do not give much meaning to the text. Here, we have removing characters apart from a-z, A-Z and 0-9. 



