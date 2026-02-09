#let title = "Project1 Statistics 01204314"
#let author = "Kongphop Tirattanaprakom 6610505250"

#set page(width: 210mm, height: 297mm, margin: (
  top: 25mm,
  bottom: 20mm,
  left: 25mm,
  right: 20mm,
))
#set text(font: "Liberation Serif", size: 12pt)

= #title

*Author:* #author

== Part 1 (Dataset A & B)

=== EDA

Firstly merge the dataset A and B together by using join on "Country Name", then drop out the columns which its missing values is more than 30% from shape (216, 198) result in shape (216, 84), but it is still having missing values. 

Separates the goal to predict out which is "Gender Ratio Class" which will be filled with mode. Although the data is a numerical data, the values should contains 5 distinct values only so mean is not the choice.

Other columns will be separated by numerical data and categorical data. For numerical data will be checked with skewness to indicate how to be filled, which if skewness is more than 0.5 (moderately or highly skewed), it will be filled with median and if it is less than 0.5 (fairly symmetric) will be filled with mean.

Finally save the data frame which is cleaned in the file "part1_cleaned.csv" to be proceed in the next file.

=== Modeling (Classification)

==== Workflow 1

Choose to list all the unique value from all columns in the data to check its type and variance of the data. 

Notes:
- In dataset A, Country Name each separate a perfectly 1:1, therefore drop the column.
- Year column also result in an only one value which is 2021, which also drop out from the data frame.
- In dataset B, Region and ThirdWorld column will be drop out because of the encoding which increase the number of features significantly.

1. Split the data into training and testing with ratio of 80:20.
2. Using smote to handling imbalance which synthetic minority oversampling. Increase the data frame shape to (355, 79).
3. Modeling with decision tree classifier model because the "Gender Ratio Class" separate distinctly into 5 different values.

Results
- Accuracy: 0.2727
- Precision: 0.1936
- Recall: 0.2197
- F1 score: 0.2022

==== Workflow 2

1. Addition the feature selection process from workflow 1, by filtering low variance. Sort the list of variance of each column, then use the minimum threshold of 200. Reduce the shape of the data frame to (355, 48).

Results
- Accuracy: 0.3864
- Precision: 0.3121
- Recall: 0.3504
- F1 score: 0.3192

==== Workflow 3

1. Use random forest regressor to selected the feature importance of the model which 23 features will be used as a selected features. The model may train with most significant features which reduce the noise and dimension.
2. Filtering the high correlation features, which "Rural population" was chosen.
3. Split the data into 80:20 and using smote to handle the imbalance plus standard scaling to scale the data.
4. Model with random forest classifier, which use the amount of 100 of decision tree. The model is selected because it does better when the model needs complicate decision compared to normal decision tree.

Results
- Accuracy: 0.5000
- Precision: 0.5405
- Recall: 0.4977
- F1 score:0.4393

==== Result interpretation

Based on my last workflow, the model perform with accuracy of fifty-fifty which doesn't produce a good prediction and also the f1 score is not even at fifty percent which indicates that the model does not perform a predictive performance.

#grid(
  columns: (auto, auto),
  column-gutter: 5em,
  row-gutter: 1em,
  align: horizon,

  align(center)[Confusion Matrix],
  [],
  $ mat(delim: "[",
    5,  0,  0,  0,  1;
    1, 11,  0,  0,  4;
    4,  2,  2,  2,  1;
    1,  1,  0,  2,  0;
    1,  0,  0,  4,  2;
  ) $,

  stack(
    spacing: 1em,
    [Accuracy: 0.5000],
    [Precision: 0.5405],
    [Recall: 0.4977],
    [F1 score:0.4393],
  )
)

From the shap value plot, the spread of bee swarm tells that "Contributing family workers, female (% of female employment) (modeled ILO estimate)" significantly affects the model. From permutation importance, "Forest area (sq. km)" is more importance feature with importance mean of 0.0977 and importance standard deviation of 0.0477 which significantly affect the "Gender Ratio Class" compared to other features.

From analysis of variance, does the mean of Region is the same for all bean Gender Ratio Class?
- H0: m1 = m2 = ... = mJ (means of each category are the same.)
- H1: At least one pair is not the same

the mean of "Region" is not the same for all bean "Gender Ratio Class" because p = 0.000469 < alpha = 0.05 which rejects the H0.

From analysis of categorical data, is bean "Gender Ratio Class" independent of "Region" category?
- H0: X and Y are independent 
- H1: X and Y are not independent

They are associate (not independent) as p = 1.6016$mu$ < alpha = 0.05 -> Reject H0

=== Feature Extraction

==== LDA

From the "Gender Ratio Class" which have 5 distinct values, therefore the LDA limited to 4. The value of each LD is 0.3656, 0.3124, 0.1811, 0.1409 in order. 

#figure(
  caption:"The value for each LDA",
  image("src/lda.png", width: 75%),
)

By choosing 3 of the LD, will represent most of the separation. Sum of the 3 LDs will explain 85.91% of the group differences. On the LD1 which represent the majority of the separation, "Population ages 0-14 (% of total population)" produce the highest variance value which is 261.05.

==== PCA

#figure(
  caption:"PCA components",
  image("src/pca.png", width: 100%),
)

By listing out 69 principal components, choosing 30 PCA would cumulate up to approximately 90.75% of the total variance in the dataset which explains the maximum of information has preserved. "Population ages 0-14 (% of total population)" produce the amount of the information in PC1 equals to 0.2107.

== Part 2 (Dataset C & B)

=== EDA

Firstly merge the dataset C and B together by using the join on "Country Name", then drop out the columns which its missing values is more than 30% from shape (1080, 197) result in shape (1080, 83), but it is still having missing values. 

All of the columns will be separated by numerical data and categorical data. For numerical data will be checked with skewness to indicate how to be filled, which if skewness is more than 0.5 (moderately or highly skewed), it will be filled with median and if it is less than 0.5 (fairly symmetric) will be filled with mean.

Finally save the data frame which is cleaned in the file "part2_cleaned.csv" to be proceed in the next file.

=== Clustering

KMeans clustering with selected k of 5 produce these performance evaluation:
- Silhouette Index:  0.6542
- Davies-Bouldin Index:  0.4363
- Dunn Index:  0.6694
- CH Index:  7176.3242

By simulates with elbow method and silhouette score, number of cluster selected is 5. In the box plots of the value in the cluster, it separates the "GNI per capita, PPP (current international \$)" into 5 groups.

#figure(
  caption: "Simulation of WCSS (elbow method) and silhouette score",
  image("src/elbow_n_silhouette.png", width: 100%),
)

Explains each cluster with respect of "GNI per capita, PPP (current international \$)"
1. Cluster 0: around 0-10000
2. Cluster 1: around 35000-50000
3. Cluster 2: around 10000-20000
4. Cluster 3: around 50000-65000
5. Cluster 4: around 20000-35000

Select these variables "Rural population (% of total population)", "Agricultural land (% of land area)", "Crop production index (2014-2016 = 100)", "Merchandise trade (% of GDP)" and "Access to electricity (% of population)". And plot in the radar graph results in the shades of gray of the data.

The cluster 0 explains that the higher the agricultural land and rural population, the crop production increases. As the access of electricity and merchandise trade is low compared with others. On the other hand, cluster 3, which merchandise trade is higher in percentage of GDP, results with a higher access of electricity but lower the crop production index which reflects the lower agricultural land and rural population. About the other clusters are the shades between these two.

=== Modeling (Regression)

==== Workflow 1

Choose to list all the unique value from all columns in the data to check its type and variance of the data. 

Notes:
- In dataset C, "Country Name" is a categorical data with null percent equals to 0, but after checking its number of unique value of 216. Dropping the column would perform better than encoding as the feature will increase too much.
- There are 2 more in categorical data which is "Region" and "ThirdWorld", after checking its number of unique value of 13 and 2 in order. Choosing to do the one-hot encoding would not increase too much dimension, the shape of the final encoding is (1080, 92).

1. Split the data into training and testing with ratio of 80:20.
2. Using standard scaling to scale the data.
3. Modeling with linear regression to perform a simple prediction.

Results
- RMSE:  7762.99
- MAE:  5420.09
- MAPE:  64.08
- R2:  0.8610

==== Workflow 2

1. Addition the filtering extreme value using cook's distance from the workflow 1, reduce the shape of the data from (1080, 92) to (1022, 92).
2. Change the model to ridge with alpha equals to 0. Use the regularization might gives a better performances.

Results
- RMSE:  6119.78
- MAE:  4425.61
- MAPE:  51.25
- R2:  0.9146

==== Workflow 3

1. Use random forest regressor to selected the feature importance of the model which 23 features will be used as a selected features. The model may train with most significant features which reduce the noise and dimension.
2. Split the data into 80:20 and use standard scaling to scale the data.
3. Model with multi-layer perceptron regressor, which the layer shape is (100, 50) and the activation function is "ReLU" with max iteration of 10000. By using complex regressor layers, it can result in a better performance as the regular linear regression might underfitting the features.

Results
- RMSE:  3821.11
- MAE:  2485.58
- MAPE:  25.25
- R2:  0.9667

==== Result interpretation

Based on my last workflow, the model perform very well on r2 score which is 0.9667, which implies that all the variability in the dependent variable is explained by the independent variables. And the root mean square error, mean absolute error and mean average percentage error improves in each workflow.

From the shap value plot, the spread of bee swarm tells that "GDP (current US\$)" significantly affects the model. From permutation importance, "Price level ratio of PPP conversion factor (GDP) to market exchange rate" is more importance feature with the importance mean of 0.08740 and importance standard deviation of 0.0109 which significantly affect the "GNI per capita, PPP (current international \$)" compared to other features.

==== Visualization

The dashboard is presenting "how education affects the GNI per capita". There are 4 slicers which are Year, Country Name, Region and ThirdWorld.

For each graphs
1. GNI per capita with respect to each 3 school enrollment layers: a distribution graph explains how many people enrollment in each level of education which separates by their country GNI.
2. Amount of countries to GNI level: a part-to-whole graph display how many countries GNI by separate GNI into each level.
3. GNI per capita with respect to each education level: correlation graphs which break graph 1 down into more detailed information separated by education level.
4. Sum of school enrollment (% gross) per year: a time series graph shows how many school enrollment each year compared with each education level.
5. Education support to GNI: a distribution graph shows how GNI acts with government expenditure on education from its total (% of GDP).

Narrative structure is partition poster which the large one in the middle explain overall of the present data, and in the next sub poster shows each details and new insight related to the main topics.
