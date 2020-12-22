# Cancer-Diagnosis-Prediction

This project was developed with the help of AppliedAI team based in India. 

Based on the given set of features such as gene, mutation and text associated with a given data, the machine learning models would predict the type of cancer that a person would have under 9 classes. 


The machine learning models that were used in the process of prediction were K Nearest Neighbors, Linear Support Vector Machines (SVM), Logistic Regression, Stacking Classifer, Voting Classifier and Random Forests Classifiers. 


Given a set of input features such as gene, the type of mutation and the text that is associated with the gene and the mutation, the machine learning model could classify the chances of a person to suffer from cancer. In addition to this, it would also give an indication of the type of class of the cancer from 1 to 9. 



We have to be looking at different metrics in the problem before diagnosing the solution. We would be looking for some metrics such as log loss, accuracy and so on for different models and add those in tables so that we would be better able to decide the right kind of algorithm for the problem. 



![](Images/2020-12-22%20(1).png)
Here, we could see that the number of points that we have taken into consideration are 3321. We have just 2 features with the file that we have taken namely the ID and the Text associated with it. We see just 5 rows just to consider the head of the data frame. 


![](Images/2020-12-22%20(2).png)
We would be preprocessing the text and converting it into a mathematical vector so that those values could be given to the machine learning model. We first take into consideration the stop words which are very important in english. Later, we would be removing every special character as they do not give much meaning to the text. Here, we are removing characters apart from a-z, A-Z and 0-9. In addition to this, we are lowecasing the words and removing all the stop words such as "the, and" and so on. We also must delete the rows that do not contain any text. The code in the second cell would do exactly do that. We would be printing the id number of the row that does not contain any text. If a row contains text, we would store those valeus in a new variable.



![](Images/2020-12-22%20(3).png)
We would be dividing the whole data into train, cross validation and test set respectively. We must ensure that the machine learning model must do well not only on the cross validation set but also the test set. Thus, we would be dividing the entire data into train, cross validation and test set. 
After dividing, we see that the total number of points on the training data are 2124. The number of data points on the test data are 665 and the points in the cross validation data is 532. 

![](Images/2020-12-22%20(5).png)
We could see from the diagram that most of the data points in the training data contain class 7 as their output. We are able to see the distribution of the data with respect to the class of cancer. 
When we see the distribution of the classes on the test data, we also see that most of the test data contain output as class 7. Therefore, histograms could be used to see how the data is distributed. 

![](Images/2020-12-22%20(7).png)
We would be first using a random model as our initial model which could be used as a benchmark for the other machine learning models. If a machine learning model has a log loss higher than the random model, it would be better not use that model as it does not even perform better than a random model which gives random set of values of y for prediction. We are also using a confusion matrix where on the y axis, we have the actual class and on the x axis, we have the predicted class. Ideally, we must have a diagonal line between the predicted class and the actual class so that there is no misclassification between classes. 
![](Images/2020-12-22%20(8).png)
We must also take into consideration precision and recall as they are some of the important metrics which we cannot take for granted. Accuracy cannot be a reliable metric to evaluate the performance of a machine learning model in classification problems. For example, if we have a test set that contains only one output (either 1 or 0) and we use a model that would return just one value without performing machine learning operations, the accuracy of the model would be high if we consider it as a metric. However, the model does not perform any machine learning operations but just returning 1 or 0. Thus, we can say that accuracy is not the best reliable metric for classification problem. After considering recall and precision, however, we could evaluate the model much better. The diagram shows precision and recall confusion matrix respectively. 
![](Images/2020-12-22%20(9).png)
We would also take a look at the index of genes that are associated with the data. Here, we find that as the index of the gene increases, their frequency decreases. We are plotting a histogram just to see how the data is spread. 
![](Images/2020-12-22%20(10)/png)
We are also plotting the cumulative distribution of genes so that we take into consideration the frequency of the genes and add them with different types of genes to get a clear picture. 


## Naive Bayes
![](Images/Naive_Bayes/2020-12-22%20(23).png)
We choose different values of alpha to check how well the model does on the test set. We can see that in the training phase, we tend to see that the model does very well for the alpha value of 0.1 and we might conclude soon that this is the best value of alpha for naive bayes. There might be chances, however, that the model is overfitting to the training data and it does not perform well on the test data as a result of high variance. Therefore, we must also check the log loss for the test set. At the bottom, we see that the best value of alpha is 0.1 as it not only reduces the log loss for the training set but also for the cross validation and the test set. By looking at the graph, therefore, we can conclude that the best value for the naive bayes classifier is 0.1 respectively. 

![](Images/Naive_Bayes/2020-12-22%20(24).png)
Now, we would be testing the model with the best hyperparameters, in our case being 0.1 for the alpha value. We use a multinomial naive bayes classifier and use the training set with the labels to it. We use predict the probabilities rather just just get a discrete 1 or a 0. We see that the log loss for the model is 1.28 (approx) respectively. We also see that the misclassification rate of the model with alpha being equal to 0.1 is 0.39 (approx). We can conclude that the model was able to predict the output for the test set accurately 60 percent of the time which is not bad when we look at the complexity of the problem at hand. 

K Nearest Neighbor



