
# Importing required libraries for data analysis and machine learning
library(tidyverse)
library(caret)
library(neuralnet)
library(corrplot)
library(gridExtra)
library(ggthemes)

# Establishing a seed to ensure reproducibility of results
set.seed(42)

# Loading the Pima Indians Diabetes dataset into the R environment
pima_dataset <- read.csv("pima-indians-diabetes.csv", header=FALSE)
names(pima_dataset) <- c("Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                         "Insulin", "BMI", "DiabetesPedigree", "Age", "DiabetesStatus")



### 1. Executive Summary
#Global Concern: Diabetes mellitus significantly impacts global health, particularly in developing countries.
#Rising Prevalence: WHO estimates indicate a sharp increase in diabetes cases worldwide, with a projection of 552 million people affected by 2030.
#Major Health Risks: Diabetes leads to serious health complications like blindness, amputation, and kidney failure.
#Need for Advanced Analysis: Leveraging computational analytics on clinical big data can enhance medical prediction and forecasting, aiding in patient-centric healthcare and cost reduction.


### 2. Exploring the Dataset

#### Initial Peek at the Data


# Displaying the top rows of the dataset for a preliminary inspection
head(pima_dataset)

# Generating a statistical summary to understand data distribution and spread
summary(pima_dataset)




#### Analyzing Outcome Distribution

# Visualizing the distribution of diabetes outcomes to understand the balance of classes
ggplot(pima_dataset, aes(DiabetesStatus, fill = as.factor(DiabetesStatus))) +
  geom_bar(colour="black") + 
  scale_fill_manual(values=c("light green", "light blue")) +
  guides(fill=FALSE) +
  ggtitle("Distribution of Diabetes Status in the Dataset")



#### Examining Age and BMI Patterns

# Categorizing Age into groups
pima_dataset <- pima_dataset %>%
  mutate(AgeGroup = cut(Age, breaks = c(0, 25, 50, 75, 100), 
                        labels = c("0-25", "26-50", "51-75", "76-100"), 
                        include.lowest = TRUE))

# Plotting Age Distribution with categorized age groups
ggplot(pima_dataset, aes(x = Age, fill = AgeGroup)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_fill_brewer(palette = "Set3") +
  ggtitle("Age Distribution in the Study Group")


# Categorizing BMI into groups
pima_dataset <- pima_dataset %>%
  mutate(BMIGroup = cut(BMI, 
                        breaks = quantile(BMI, probs = seq(0, 1, length.out = 31), na.rm = TRUE), 
                        include.lowest = TRUE, 
                        labels = FALSE))

# Plotting BMI Distribution with categorized BMI groups
ggplot(pima_dataset, aes(x = BMI, fill = as.factor(BMIGroup))) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_fill_viridis_d() +
  ggtitle("BMI Distribution Among Participants")






### 3. Constructing and Training the Neural Network

#### Data Split for Training and Validation

# Dividing the dataset into training and validation sets for model building
data_partition_index <- createDataPartition(y = pima_dataset$DiabetesStatus, times = 1, p = 0.1, list = FALSE)
training_data <- pima_dataset[-data_partition_index,]
validation_data <- pima_dataset[data_partition_index,]

# Formulating the neural network structure based on the dataset
formula_for_nn <- as.formula(paste("DiabetesStatus ~", paste(names(training_data)[!names(training_data) %in% "DiabetesStatus"], collapse = " + ")))

################################
# Check if all columns in the formula are in the dataset
all_variables <- all(names(training_data) %in% all.vars(formula_for_nn))
print(all_variables)

# If 'all_variables' is FALSE, update the formula to match the dataset
# For example:
# formula_for_nn <- as.formula("DiabetesStatus ~ .", data = training_data)

# Explicitly define the formula
formula_for_nn <- as.formula("DiabetesStatus ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigree + Age")

# Check structure of training_data
print(str(training_data))

# Train the neural network with the explicitly defined formula
nn_model <- neuralnet(formula_for_nn, data=training_data, hidden=c(8), linear.output=FALSE, threshold = 0.06)
prediction_results <- compute(nn_model, validation_data[,1:8])

##################################



#### Initiating Neural Network Training

# Training the neural network with specified architecture
nn_model <- neuralnet(formula_for_nn, data=training_data, hidden=c(8), linear.output=FALSE, threshold = 0.06)
prediction_results <- compute(nn_model, validation_data[,1:8])




### 4. Assessing Model Performance

#### Generating a Confusion Matrix

# Creating a confusion matrix to evaluate model accuracy and precision
confusion_matrix_result <- confusionMatrix(as.factor(round(prediction_results$net.result)), 
                                           as.factor(validation_data$DiabetesStatus))
confusion_matrix_result



#### Visual Comparison of Predictions and Actual Outcomes

# Plotting to compare the model's predictions against actual outcomes
plot(validation_data$DiabetesStatus, prediction_results$net.result, col='blue', pch=16, 
     ylab = "Neural Network Predictions", xlab = "Actual Diabetes Status")
abline(0,1)



#### Calculating the Root Mean Square Error

# Computing the RMSE to measure the prediction error of the model
rmse_metric <- sqrt(mean((validation_data$DiabetesStatus - prediction_results$net.result)^2))
rmse_metric




# RMSE is a measure of the differences between predicted and observed values
# Here, we calculate RMSE to quantify the accuracy of our neural network model

# Squaring the difference between observed and predicted, averaging, then taking the square root
rmse_score <- sqrt(mean((validation_data$DiabetesStatus - prediction_results$net.result)^2))

# Displaying the RMSE value
rmse_score




### 5. Conclusion and Reflections

#Conclusion: Understanding Our Findings
#What We Discovered
#We've completed our analysis using a neural network to predict diabetes outcomes. The key takeaway is the RMSE score we calculated. This score tells us how accurate our predictions were compared to the actual data. Think of it as a way to measure how close our model's guesses were to the real answers.

#How Good Was Our Model?
#The RMSE value helps us understand how well our model did. A lower RMSE means our predictions were more accurate. We can compare this score to other models or benchmarks to see if our model is doing well. If our RMSE is lower than others, it means our model is quite good at predicting who might have diabetes.

#Looking Ahead
#There's always room to make our model better. We could try different ways of setting up the neural network or use more data for training. We can also explore new methods that might make our predictions even more accurate. All of this can help us build a tool that's really useful for healthcare, especially for understanding and predicting diabetes.
