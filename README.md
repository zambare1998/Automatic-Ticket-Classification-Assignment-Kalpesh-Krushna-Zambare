# Automatic-Ticket-Classification-Assignment-Kalpesh-Krushna-Zambare
Case Study: Automatic Ticket Classification


# OVERALL CONCLUSION OF ASSIGNMENT

* The "Automatic Ticket Classification" assignment is a practical application of text classification techniques in the domain of customer support and complaint management. It involves the development of machine learning models to automatically categorize customer complaints into relevant topics or categories. This assignment is particularly relevant in scenarios where organizations receive a large volume of customer complaints and need an efficient way to route them to the appropriate teams or departments for resolution.

* The "Automatic Ticket Classification" assignment is a valuable exercise that demonstrates the application of natural language processing (NLP) and machine learning techniques in solving real-world business challenges related to customer support and complaint management. It showcases how technology can be leveraged to enhance customer service, improve efficiency, and streamline complaint resolution processes.






Here's an overview and detailed conclusion of our "**Automatic Ticket Classification**" assignment:

### **Assignment Overview:**

The "Automatic Ticket Classification" assignment involved the following key steps and tasks:

1.  **Data Preprocessing**:
  *  We began by cleaning and preprocessing the dataset containing customer complaints. This included tasks such as text normalization, removal of stopwords, and lemmatization. Additionally, we labeled each complaint with a topic category.

2.  **Data Exploration**: You explored the dataset to understand its distribution across different complaint topics. This involved visualizations and statistical analysis to gain insights into the data.

3.  **Text Vectorization**: To prepare the text data for machine learning, we used the Count Vectorizer and TF-IDF Transformer to convert text complaints into numerical features.

4.  **Model Selection**: We selected three different models for classification:

    * **Multinomial Naive Bayes**
    * **Logistic Regression**
    * **Decision Tree**

5.  **Model Building and Evaluation:**

   * We trained each of the selected models on the training data.
   * Hyperparameter tuning was performed to find the best parameters for each model.
   * The F1 Score, a weighted average of precision and recall, was used as the evaluation metric for model performance.

6. **Model Comparison**: You compared the F1 Scores of the three models to determine which one performed the best on the test data.

7. **Model Saving**: The best-performing model, which was the Logistic Regression model, was saved as a pickle file for future use.

8.  **Custom Text Prediction**: Finally, we created a function to predict the topic of a custom text complaint using the saved Logistic Regression model.


### **Detailed Conclusion:**

1.  **Data Preprocessing**: Data preprocessing is a crucial step in natural language processing (NLP) tasks. By cleaning and preprocessing the customer complaints, you ensured that the text data was in a suitable format for machine learning. Text normalization, removal of stopwords, and lemmatization improved the quality of the data.

2.  **Model Selection and Training**: We experimented with three different classification models: Multinomial Naive Bayes, Logistic Regression, and Decision Tree. Each model was trained on the preprocessed training data.

3.  **Hyperparameter Tuning**: Hyperparameter tuning helps in finding the best combination of hyperparameters for each model. Grid search with cross-validation was used to optimize hyperparameters, resulting in improved model performance.

4.  **Model Evaluation**: To evaluate the models, we used the F1 Score, which balances precision and recall. This metric is particularly useful in cases where class imbalance exists, as it provides a single score that considers both false positives and false negatives.

5.  **Best Model Selection**: The Logistic Regression model outperformed the other models with the highest F1 Score of 0.94 on the test data. This makes it the best model for classifying customer complaints into relevant topics.

6.  **Model Saving**: The Logistic Regression model was saved as a pickle file, allowing for easy reusability without the need for retraining.

7.  **Custom Text Prediction**: A custom text prediction function was created to demonstrate the practical use of the best model. Users can input their complaints, and the model will predict the most relevant topic.


### **Why Logistic Regression is the Best Model:**

**Logistic Regression proved to be the best model for this task for several reasons:**

* **High F1 Score**: It achieved the highest F1 Score of 0.94, indicating excellent precision and recall on the test data.

*  **Interpretability**: Logistic Regression provides interpretability, making it easier to understand the factors influencing the predictions.

* **Efficiency**: It is computationally efficient and can handle large datasets.

* **Robustness**: Logistic Regression performed well even with the text data after preprocessing.

* **Consistency**: The model provided consistent and reliable predictions.


### **Overall Benefits:**

*  The assignment demonstrated the entire workflow of text classification, from data preprocessing to model evaluation.

*  It showcased the importance of choosing the right evaluation metric, especially when dealing with imbalanced datasets.

* The custom text prediction function adds practical value, allowing users to classify new complaints into relevant topics.

*  The Logistic Regression model, with its high F1 Score, can be deployed in real-world scenarios to automate the classification of customer complaints, improving efficiency and customer service.


*In conclusion, the "**Automatic Ticket Classification**" assignment successfully achieved its objectives, and the Logistic Regression model emerged as the best choice for accurately classifying customer complaints into topics. This assignment provides valuable insights into text classification and its practical applications.*

