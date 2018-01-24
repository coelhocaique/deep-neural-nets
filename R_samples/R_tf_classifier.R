#http://upflow.co/l/W6Ql/data-science-2/deep-learning-tensorflow-r-tutorial/utm_campaign=step-by-step-tutorial-deep-learning-with-tensorflow-in-r

devtools::install_github("rstudio/tfestimators")
library(tfestimators)
library(readr)
library(dplyr)

install_tensorflow()

#> Error: Prerequisites for installing
#> TensorFlow not available.  Execute the
#> following at a terminal to install the
#> prerequisites: $ sudo
#> /usr/local/bin/pip install --upgrade
#> virtualenv

donor_data <- read_csv("https://www.dropbox.com/s/ntd5tbhr7fxmrr4/DonorSampleDataCleaned.csv?raw=1")

glimpse(donor_data)

# TensorFlow library doesn’t tolerate missing values, therefore,
# we will replace missing factor values with modes and missing numeric values with medians.

# function copied from
# https://stackoverflow.com/a/8189441/934898
my_mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}

donor_data <- donor_data %>%
  mutate_if(is.numeric,
            .funs = funs(
              ifelse(is.na(.),
                     median(., na.rm = TRUE),
                     .))) %>%
  mutate_if(is.character,
            .funs = funs(
              ifelse(is.na(.),
                     my_mode(.),
                     .)))

predictor_cols <- c("MARITAL_STATUS", "GENDER",
                   "ALUMNUS_IND", "PARENT_IND",
                   "WEALTH_RATING", "PREF_ADDRESS_TYPE")

# Convert feature to factor
donor_data <- mutate_at(donor_data,
                       .vars = predictor_cols,
                       .funs = as.factor)

feature_cols <- feature_columns(
  column_indicator(
   column_categorical_with_vocabulary_list(
     "MARITAL_STATUS",
     vocabulary_list = unique(donor_data$MARITAL_STATUS))),
   column_indicator(
     column_categorical_with_vocabulary_list(
       "GENDER",
       vocabulary_list = unique(donor_data$GENDER))),
   column_indicator(
     column_categorical_with_vocabulary_list(
       "ALUMNUS_IND",
       vocabulary_list = unique(donor_data$ALUMNUS_IND))),
   column_indicator(
     column_categorical_with_vocabulary_list(
       "PARENT_IND",
       vocabulary_list = unique(donor_data$PARENT_IND))),
   column_indicator(
     column_categorical_with_vocabulary_list(
       "WEALTH_RATING",
       vocabulary_list = unique(donor_data$WEALTH_RATING))),
   column_indicator(
     column_categorical_with_vocabulary_list(
       "PREF_ADDRESS_TYPE",
       vocabulary_list = unique(donor_data$PREF_ADDRESS_TYPE))),
   column_numeric("AGE"))

row_indices <- sample(1:nrow(donor_data),size = 0.8 * nrow(donor_data))
donor_data_train <- donor_data[row_indices, ]
donor_data_test <- donor_data[-row_indices, ]

donor_pred_fn <- function(data) {
    input_fn(data,features = c("AGE", "MARITAL_STATUS",
                          "GENDER", "ALUMNUS_IND",
                          "PARENT_IND", "WEALTH_RATING",
                          "PREF_ADDRESS_TYPE"),
             response = "DONOR_IND")
}

classifier <- dnn_classifier(
  feature_columns = feature_cols,
  hidden_units = c(80, 40, 30),
  n_classes = 2,
  label_vocabulary = c("N", "Y"))

train(classifier,input_fn = donor_pred_fn(donor_data_train))

predictions_test <- predict(
  classifier,
  input_fn = donor_pred_fn(donor_data_test))
predictions_all <- predict(
  classifier,
  input_fn = donor_pred_fn(donor_data))


evaluation_test <- evaluate(
  classifier,
  input_fn = donor_pred_fn(donor_data_test))
evaluation_all <- evaluate(
  classifier,
  input_fn = donor_pred_fn(donor_data))
