#Opening required packages:
library(dplyr)
library(caret)
library(car)
library(ggplot2)
library(tidyr)
library(randomForest)

#Importing the data:
file_path = "/Users/stevenleicht/Desktop/personal_health_data.csv"
data = read.csv(file_path)

#Selecting the data I want to analyze:
selected_data = data %>%
  select(day, body_weight, sleep_quantity_apple, daily_steps_apple, lowest_hr_apple, sleep_quantity_oura,
         sleep_score_oura, lowest_hr_oura, daily_steps_oura, calories_burned_oura, activity_score_oura,
         calories_burned_apple, readiness_score_oura, calories_consumed)

#Running a correlation matrix:
correlation_matrix = cor(selected_data)
print(correlation_matrix)
#Highlights:
#There's a strong positive correlation between body weight with Apple's sleep quantity (0.509), Oura's sleep quantity (0.543), and Oura's readiness score (0.586). This suggests that as body weight increases, so does sleep duration and daily readiness score
#There's also a strong positive correlation between calories burned and daily steps taken (Apple = 0.756; Oura = 0.650). This suggests that as daily steps taken increases, so does the number of calories burned
#There's a very strong positive correlation between sleep quantity for Apple and Oura (0.978)
#There's a very strong positive correlation between Oura's sleep score and Apple's sleep quantity (0.875)
#There's a moderate negative correlation between Oura's lowest heart rate and Apple's sleep quantity (-0.412). This suggests that as sleep duration increases, the lowest heart rate decreases (and vice versa)
#There's also a moderately negative correlation between Oura's lowest heart rate and Oura's readiness score (-0.542). This suggests that as the readiness score increases, the lowest heart rate decreases (and vice versa)
#There's a weak negative correlation between body weight with Apple's daily steps taken (-0.069) and calories consumed (-0.051). This suggests that daily steps taken and calorie consumption do not have a significant effect on body weight

#Creating a linear regression model to compare sleep, activity, and nutrition metrics (across Apple and Oura) to body weight:
lr_model = lm(body_weight ~ sleep_quantity_apple + daily_steps_apple + lowest_hr_apple +
                sleep_quantity_oura + sleep_score_oura + lowest_hr_oura + daily_steps_oura +
                calories_burned_oura + activity_score_oura + calories_burned_apple +
                readiness_score_oura + calories_consumed, data = selected_data)
summary(lr_model)
#There's a residual standard error of 0.7333 lbs on 17 degrees of freedom
#There's an adjusted R-squared value of 0.1731. This means that there is not a strong relationship (or the model is over-fitting) between body weight and the selected health metrics
#The p-value is 0.2143. This also means that these health metrics do not have a significant effect on body weight

#Checking the assumptions of the model:

#Finding the residuals from the linear regression model:
pca_residuals = residuals(lr_model)

#Looking at the residuals of the model via a Q-Q Plot:
qqnorm(residuals(lr_model), main = "QQ Plot of Residuals")
qqline(residuals(lr_model), col = "red")
#The line is relatively straight, meaning that there is a normal distribution across the data/model

#Running an ANOVA to further evaluate the model:
anova_result = anova(lr_model)
print(anova_result)
#A high F-value (9.106) and a low p-value (< 0.01) between body weight and Apple's sleep quantity shows that there's a strong relationship between the two variables
#A relatively high F-value (5.123) and a low p-value (< 0.05) between body weight and Oura's calories burned also shows that there's a strong relationship between the two variables
#All other variables have low F-values and a p-value > 0.05... meaning that they do not show significant correlations to body weight
#The residual sum of squares = 9.1415 lbs and the mean square for residuals = 0.5377 lbs... which both showcase the average unexplained variance between body weight and all other health variables

#Performing a 10-fold cross-validation on the linear regression model to dismiss over-fitting and check its validity:

#Setting the seed for proper reproducibility purposes:
set.seed(123)

#Splitting the data for proper 10-fold cross-validation:
train_control = trainControl(method = "cv", number = 10, verboseIter = TRUE)

#Creating the 10-fold CV model:
cv_model = train(body_weight ~ sleep_quantity_apple + daily_steps_apple + lowest_hr_apple +
                   sleep_quantity_oura + sleep_score_oura + lowest_hr_oura + daily_steps_oura +
                   calories_burned_oura + activity_score_oura + calories_burned_apple +
                   readiness_score_oura + calories_consumed, data = selected_data,
                 method = "lm", trControl = train_control)
print(cv_model)
#The root mean squared error (RMSE) = 0.9331... meaning that there is a 0.9331 lb average deviation difference between the actual and predicted values of body weight
#The R-squared value is 0.7120... meaning that the model performs well and is not over-fitting
#The mean absolute error (MAE) = 0.8502... meaning that there's a 0.8502 lb average absolute difference between the actual and predicted values of body weight

#Checking the linear regression model for multicollinearity:
vif(lr_model)
#VIF values higher than 10 show signs of multicollinearity. For this model - the variables included are Apple's sleep quantity, Apple's daily steps taken, Oura's sleep quantity, Oura's daily steps taken, Oura's calories burned and Apple's calories burned
#VIF values between 5 and 10 show moderate multicollinearity. For this model - the variables included are Oura's sleep and readiness scores
#VIF values less than 5 don't show multicollinearity. For this model - the variables included are Apple and Oura's lowest heart rate, Oura's activity score, and calories consumed

#Performing a principal component analysis (PCA) to remove multicollinearity from the model and further evaluate the correlations between body weight and the selected metrics:
pca_result = prcomp(selected_data[, c("body_weight", "sleep_quantity_apple", "daily_steps_apple",
                                      "lowest_hr_apple", "sleep_quantity_oura", "sleep_score_oura",
                                      "lowest_hr_oura", "daily_steps_oura", "calories_burned_oura",
                                      "activity_score_oura", "calories_burned_apple", "readiness_score_oura",
                                      "calories_consumed")], center = TRUE, scale. = TRUE)
summary(pca_result)
#PC1-PC6 explain the most variance (90.60%), with PC1 explaining the most

#Visualizing the PCA through a Scree Plot:
screeplot(pca_result, main = "Scree Plot")

#Calculating cumulative variance to help decide the correct number of principal components to keep/analyze:
cumulative_variance = cumsum(summary(pca_result)$importance[2,])
plot(cumulative_variance, type = "b", xlab = "Number of Components", ylab = "Cumulative Variance Explained")

#I will only evaluate the first six principal components since they explain the vast majority of the variance within the data

#Selecting the number of principal components I want to use:
num_components = 6
r_pca_data = pca_result$x[, 1:num_components]

#Adding the dependent variable to the PCA data set to create another linear regression model:
pca_results = pca_result$x
pca_data_w_dv = as.data.frame(pca_results)
pca_data_w_dv$body_weight = selected_data$body_weight

#Creating another linear regression model based on the PCA data:
pca_lr_model = lm(body_weight ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6, data = as.data.frame(pca_data_w_dv))
summary(pca_lr_model)
#There's a residual standard error of 0.3294 lbs on 23 degrees of freedom
#There's an adjusted R-squared value of 0.8331... meaning that the model is very strong and not over-fitting
#The p-value is < 0.001... meaning that these chosen sleep/activity metrics impact body weight

#Checking the assumptions of the PCA model using a Q-Q Plot:

#Finding the residuals from the PCA model:
pca_residuals = residuals(pca_lr_model)

#Creating the Q-Q Plot:
qqnorm(pca_residuals)
qqline(pca_residuals, col = "red")
#There's some skewness towards the bottom of the graph, but the rest of the data is normally distributed

#Running an ANOVA for the PCA model:
pca_anova_result = anova(pca_lr_model)
print(pca_anova_result)
#PC1, PC2, PC5, and PC6 all have high F-values and very low p-values (< 0.001)... meaning that they are highly significant predictors of body weight
#PC3 and PC4 have low F-values and high p-values (> 0.05)... meaning that they are not significant predictors of body weight

#Checking the validity of the PCA model:

#Performing cross-validation of the PCA model to further check its validity:

#Setting the seed for proper reproducibility purposes:
set.seed(123)

#Splitting the data for proper 10-fold cross-validation:
pca_train_control = trainControl(method = "cv", number = 10, verboseIter = TRUE)

#Creating the 10-fold cross-validation model for the PCA:
pca_cv_model = train(body_weight ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6,
                     data = pca_data_w_dv, method = "lm", trControl = pca_train_control)
print(pca_cv_model)
#The root mean squared error (RMSE) = 0.4087... meaning that there is a 0.4087 lb average deviation difference between the actual and predicted values of body weight
#The R-squared value is 0.7601... meaning that the model performs well and is not over-fitting
#The mean absolute error (MAE) = 0.3591... meaning that there's a 0.3591 lb average absolute difference between the actual and predicted values of body weight

#Checking the PCA model for multicollinearity:
vif(pca_lr_model)
#As expected, there's no multicollinearity in the PCA model

#Creating a random forest model from the PCA linear regression model to determine which principal components are most important:
set.seed(123)
rf_data = pca_data_w_dv
rf_model = randomForest(body_weight ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6,
                        data = rf_data, ntree = 500, mtry = 2, importance = TRUE)
print(rf_model)
#The mean of squared residuals (MSE) = 0.3816 lbs
#The percentage of variance explained = 39.29%

#Creating an importance model to further determine which principal components are most important:
importance_scores = importance(rf_model)
print(importance_scores)

#Visualizing the importance of each principal component:

#Converting importance scores to a data frame for plotting
importance_df = as.data.frame(importance_scores)
importance_df$Variable = rownames(importance_df)

#Bar plot explaining the importance of each principal component/variable:
ggplot(importance_df, aes(x = reorder(Variable, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Principal Component", y = "Importance (IncNodePurity)")
#PC2 is the most and PC4 is the least important principal component

#Creating another linear regression model eliminating PC3 and PC4 based on the conclusions of the cross-validation and random forest models:
pca_lr_model_2 = lm(body_weight ~ PC1 + PC2 + PC5 + PC6, data = as.data.frame(pca_data_w_dv))
summary(pca_lr_model_2)
#There's a residual standard error of 0.3182 lbs on 25 degrees of freedom
#There's an adjusted R-squared value of 0.8443. This is the highest adjusted R-squared value of the three models
#There's a p-value < 0.001... this model is also highly significant in showing the correlation between body weight and specific sleep/activity metrics

#Checking the assumptions of the updated PCA model using a Q-Q Plot:

#Finding the residuals from the updated PCA model:
pca_residuals_2 = residuals(pca_lr_model_2)

#Creating the Q-Q Plot:
qqnorm(pca_residuals_2)
qqline(pca_residuals_2, col = "red")
#Similarly to the residuals of the first PCA model, there's some skewness towards the bottom of the graph, but the rest of the data is normally distributed

#Running an ANOVA for the updated PCA model:
pca_anova_result_2 = anova(pca_lr_model_2)
print(pca_anova_result_2)
#Similarly to the first PCA model's ANOVA, each principal component has a very high F-value and a very low p-value (< 0.001)... meaning that they are all highly significant predictors of body weight

#Checking the validity of the updated PCA model:

#Performing cross-validation of the updated PCA model to further check its validity:

#Setting the seed for proper reproducibility purposes:
set.seed(123)

#Splitting the data for proper 10-fold cross-validation:
pca_train_control_2 = trainControl(method = "cv", number = 10, verboseIter = TRUE)

#Creating the 10-fold cross-validation model for the updated PCA model:
pca_cv_model_2 = train(body_weight ~ PC1 + PC2 + PC5 + PC6,
                     data = pca_data_w_dv, method = "lm", trControl = pca_train_control_2)
print(pca_cv_model_2)
#The root mean squared error (RMSE) = 0.3621... meaning that there is a 0.3621 lb average deviation difference between the actual and predicted values of body weight
#The R-squared value is 0.7876... meaning that the model performs well and is not over-fitting
#The mean absolute error (MAE) = 0.3174... meaning that there's a 0.3174 lb average absolute difference between the actual and predicted values of body weight
#Each value performs slightly better than the first PCA model, which means it is slightly more valid in determining/predicting body weight

#Checking the updated PCA model for multicollinearity:
vif(pca_lr_model_2)
#As expected, again, there's no multicollinearity in the updated PCA model

#Creating predictive values for the updated PCA model and comparing it to the actual values:

#Creating PCA predicted values:
predicted_pca_values = predict(pca_lr_model_2, newdata = pca_data_w_dv)

#Comparing the actual and PCA predictive values:
comparison_pca_data = data.frame(Actual = pca_data_w_dv$body_weight, Predicted = predicted_pca_values)
print(comparison_pca_data)

#Creating a separate data frame for visualization purposes:
comparison_pca_df = data.frame(Actual = pca_data_w_dv$body_weight, Predicted = predicted_pca_values)

#Changing the data to long format to differentiate the actual and predicted values:
comparison_pca_df_lf = comparison_pca_df %>%
  pivot_longer(cols = c(Actual, Predicted), names_to = "value_type", values_to = "body_weight")

#Visualizing the differences between the actual and PCA predicted values:
ggplot(comparison_pca_df_lf, aes(x = body_weight, y = body_weight, color = value_type)) +
  geom_point(size = 3) +
  geom_abline(intercept = 0, slope = 1, color = "green", linetype = "dashed", linewidth = 1.2) +
  labs(title = "Actual vs. PCA Predicted Body Weight",
       x = "Actual Body Weight (lbs)",
       y = "PCA Predicted Body Weight (lbs)") +
  theme_minimal()
#As seen in this visual, the actual and predicted values for body weight are nearly identical... showcasing the strength of the updated PCA model in terms of its prediction of body weight

#Conclusion:
#The selected sleep, nutrition, and activity metrics impact and are impacted by body weight
#The final PCA model performed the best in terms of correlation and accurately predicting body weight based on the chosen independent variables. This model should be used when analyzing these types of metrics/variables
#When determining the optimal body weight for an individual, whether for performance or health, looking at these metrics can be important for both research and application purposes
#Future research should dive deeper into these metrics
#Future projects involving this data can look at comparisons between wearable devices, further creations of predictive modeling, and so forth











