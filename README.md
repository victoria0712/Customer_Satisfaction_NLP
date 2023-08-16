# Customer Satisfaction NLP

This project aims to identify customer satisfaction levels using Natural Language Processing (NLP) techniques. Due to confidentiality, customer data has been replaced with a similar dataset from Kaggle. The primary goal is to categorize customer sentiments based on their feedback ('Summary'), utilizing a discrete classification of the 'Rate' scale into ['negative', 'neutral', 'positive'].

## Business Impact:

This project provides businesses with a powerful tool to assess customer sentiment, enabling data-driven decision-making for enhancing customer service.

## Dataset

The dataset consists of customer reviews and ratings, with 5 labels ranging from 1 (least satisfied) to 5 (most satisfied). The dataset contains 685 rows of data.

## Methodology

To achieve effective result summarization and communication, the 'Rate' scale has been discretely categorized into three distinct classes: 'negative', 'neutral', and 'positive'. This categorization allows for streamlined interpretation of the results and promotes response consistency.

The BERT-based model 'bert-base-cased' is employed for sentiment analysis. Using 'bert-base-cased' is motivated by the consideration that capitalized letters convey stronger tones in customer reviews.

## Hyperparameters

Given the relatively small dataset, the following hyperparameters were utilized for training the model:

- Batch Size: 8
- Learning Rate: 1e-5 (0.00001)
- Number of Epochs: 10

## Results

The trained model achieved the following performance on the test set:


          precision    recall  f1-score   support

negative       0.71      1.00      0.83        10
 neutral       0.00      0.00      0.00         3
positive       0.99      0.98      0.98        90

accuracy                           0.95       103

## Conclusion

The results demonstrate the effectiveness of the BERT-based model in categorizing customer sentiments. The high accuracy and precision for positive and negative sentiments highlight the model's capability to discern customer satisfaction levels. The project underscores the potential of NLP techniques in analyzing customer feedback and deriving valuable insights.
