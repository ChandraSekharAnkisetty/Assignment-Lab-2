library(dplyr)
library(data.tree)
library(randomForest)
library(caret) # For confusion matrix

# Load data
url <- "https://raw.githubusercontent.com/nasimm48/machine-learning/main/lab-2/data/oulad-assessments.csv"
data <- read.csv(url)

# Remove rows with NA values
data <- na.omit(data)

# Convert factors
data$code_module <- as.factor(data$code_module)
data$code_presentation <- as.factor(data$code_presentation)
data$assessment_type <- as.factor(data$assessment_type)

# Assuming 'score' needs to be converted to a categorical factor (e.g., High, Medium, Low)
# This is an example of binning the scores into three levels for classification
data$score <- cut(data$score, breaks=quantile(data$score, probs=0:3/3, na.rm=TRUE), include.lowest=TRUE, labels=c("Low", "Medium", "High"))
data$score <- as.factor(data$score)

set.seed(123)  # for reproducibility

data_set_size <- floor(nrow(data) * 0.80)
index <- sample(1:nrow(data), size = data_set_size)

training <- data[index, ]
testing <- data[-index, ]

# Fit random forest model for classification
rf <- randomForest(score ~ code_module + code_presentation + assessment_type, data = training)

# Prediction and Result data frame
predictions <- predict(rf, newdata = testing, type = "response")
result <- data.frame(Actual = testing$score, Predicted = predictions)

# Generate and print confusion matrix
confusion_matrix <- confusionMatrix(data = result$Predicted, reference = result$Actual)
print(confusion_matrix)

# Optional: Create a graphical representation of the confusion matrix
library(ggplot2)
confusion_df <- as.data.frame(confusion_matrix$table)
confusion_plot <- ggplot(confusion_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1.5, color = "red") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  ggtitle("Confusion Matrix") +
  xlab("Actual Class") +
  ylab("Predicted Class")
print(confusion_plot)
confusion_matrix
