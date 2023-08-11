# ========== Azure ML Setup ==========


# To install a list of packages
%pip install -U -r requirements.txt

# To create a Azure Machine Learning Client
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
ml_client = MLClient(credential=credential,
                     subscription_id="f49ecf56-61d3-4ccf-8ce0-431494d09f42",
                     resource_group_name="artemis-k8s",
                     workspace_name="automl-dev",
)

# to create a data store (run only once)
'''
from azure.ai.ml.entities import AzureBlobDatastore
from azure.ai.ml.entities import SasTokenConfiguration 
from azure.ai.ml import MLClient

store = AzureBlobDatastore(
    name="victorialystore",
    description="Datastore pointing to a blob container using SAS token.",
    account_name="automldev1554717007",
    container_name="victorialystore",
    credentials=SasTokenConfiguration(
        sas_token= "sp=racwdli&st=2023-05-09T06:08:46Z&se=2024-09-05T14:08:46Z&spr=https&sv=2022-11-02&sr=c&sig=QHEPHmgvR6%2BPX%2FynLatlSS4CVwi%2Fel4EXYezo%2Fn%2BELE%3D"
    ),
)
ml_client.create_or_update(store)
'''

# instantiate file system using following URI
from azureml.fsspec import AzureMachineLearningFileSystem
fs = AzureMachineLearningFileSystem('azureml://subscriptions/f49ecf56-61d3-4ccf-8ce0-431494d09f42/resourcegroups/artemis-k8s/workspaces/automl-dev/datastores/victorialystore')
fs.ls()
# refer to Microsoft Doc for more details https://learn.microsoft.com/en-us/azure/machine-learning/how-to-access-data-interactive?view=azureml-api-2&tabs=adls

# access data from data store
import pandas as pd
df = pd.read_csv("azureml://subscriptions/f49ecf56-61d3-4ccf-8ce0-431494d09f42/resourcegroups/artemis-k8s/workspaces/automl-dev/datastores/victorialystore/paths/bert/Sampledata.csv", encoding='ISO-8859-1')
df.head()

# model required numpy 1.23.5 version
import numpy as np
print(np.__version__)


# ========== BERT Setup ==========

# Setup & Config
import transformers
from transformers import BertModel, BertTokenizer, AdamW ,  get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ========== Data Exploration ==========


df.shape
df.info()

# drop missing data
missing_count = df['Verbatim'].isna().sum()
print("missing data: " + str(missing_count))
df = df.dropna(subset=['Verbatim'])
df = df[df != 'N_A'].dropna(subset=['CaseCategory'])
records = df.shape[0]
print("number of data: " + str(records))

# visualize the case category
sns.countplot(x=df.CaseCategory)
plt.xlabel('Case Category')

# visualize the review score
sns.countplot(x=df.Score)
plt.xlabel('review score')

# Convert the label to 7 categories: 'class 6', 'class 5', 'class 4','class 3','class 2','class 1','class 0'
# Higher class shows greater satisfaction
def to_sentiment(row):
  if row['CaseCategory'] == 'Compliment':
    if row['Score'] == 1:
      return 6
    elif row['Score'] == 0.75:
      return 5
    elif row['Score'] == 0.5:
      return 4
    elif row['Score'] == 0.25:
      return 3
  elif row['CaseCategory'] == 'Feedback':
    return 2
  elif row['CaseCategory'] == 'Suggestion':
    return 1
  elif row['CaseCategory'] == 'Complaint':
    return 0

df['sentiment'] = df.apply(lambda row: to_sentiment(row), axis=1)

# plot sentiment
sns.countplot(x=df.sentiment)
plt.xlabel('review sentiment')
df = df.rename(columns={'Verbatim': 'content'})
class_names = ['class 6', 'class 5', 'class 4','class 3','class 2','class 1','class 0']

ax = sns.countplot(x=df.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names)

df.head()


# ========== Data Preprocessing ==========

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# special token
tokenizer.sep_token, tokenizer.sep_token_id
tokenizer.cls_token, tokenizer.cls_token_id
tokenizer.pad_token, tokenizer.pad_token_id
tokenizer.unk_token, tokenizer.unk_token_id

# sequence length
token_lens = []

for txt in df.content:
  tokens = tokenizer.encode(txt, max_length=512, truncation=True) # without padding; padding=False
  token_lens.append(len(tokens))

sns.distplot(token_lens)  # or use sns.histplot()
plt.xlim([0, 256])
plt.xlabel('Token count')

MAX_LEN = 120

# create pytorch dataset
class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews    # df.content
    self.targets = targets    # df.sentiment
    self.tokenizer = tokenizer
    self.max_len = max_len    # 120
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])    # E.g., "Thank you very much! Phew. You are a saver . Thanks again  have a good day! Thumbs Up!"
    target = self.targets[item]         # E.g., sentiment - 3 (compliment - score 0.25)

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      truncation=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,                                    # "Thank you very much! Phew. You are a saver . Thanks again  have a good day! Thumbs Up!"
      'input_ids': encoding['input_ids'].flatten(),             # [  101,  4514,  1128,  1304,  1277,   106,  7642,  5773,   119,  1192,
                                                                #    1132,   170,  3277,  1197,   119,  5749,  1254,  1138,   170,  1363,
                                                                #    1285,   106,   157, 21631,  4832,  3725,   106,   102,     0,     0,
                                                                #       0,     0]
      'attention_mask': encoding['attention_mask'].flatten(),   # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                #   1, 1, 1, 1, 0, 0, 0, 0]]
      'targets': torch.tensor(target, dtype=torch.long)         # 3
    }
  
# Split train and test set
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['sentiment'])
# Find categories with only one data point
single_categories = df_test['sentiment'].value_counts()[df_test['sentiment'].value_counts() == 1].index.tolist()
# Move one instance from each single category to the test/validation set
for category in single_categories:
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    instance_index = df_train[df_train['sentiment'] == category].index[0]
    row_to_concat = df_train.iloc[instance_index]
    df_test = pd.concat([df_test, row_to_concat.to_frame().T], axis=0, ignore_index=True)
    df_train = df_train.drop(df.index[instance_index])
# Split test and validation set
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED, stratify=df_test['sentiment'])
#0.6, 0.3, 0.1


sorted_value_counts_df = pd.DataFrame({
    'df': df['sentiment'].value_counts().sort_index(ascending=True),
    'df_train': df_train['sentiment'].value_counts().sort_index(ascending=True),
    'df_test': df_test['sentiment'].value_counts().sort_index(ascending=True),
    'df_val': df_val['sentiment'].value_counts().sort_index(ascending=True)
})

sorted_value_counts_df

df_train.shape, df_val.shape, df_test.shape

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle = True):  # df is set to df_train, df_val, or df_test
  ds = GPReviewDataset(
    reviews=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len,
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle = shuffle,
    #num_workers=4 #  process hangs with num_workers=4
    num_workers=0  #  num_workers: how many subprocesses to use for data loading. 0 means that the data will be 
                   #  loaded in the main process. (default: 0)
  )

BATCH_SIZE = 8 #16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)  # MAX_LEN is 120
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))  # first batch from train data
data.keys()

print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


# ========== Sentiment Classification with BERT and Hugging Face ==========

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)  # E.g., hidden_size: 768, n_classes: 3
  
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(outputs.pooler_output)
    return self.out(output)


model = SentimentClassifier(len(class_names))   # class_names = ['class 6', 'class 5', 'class 4','class 3','class 2','class 1','class 0']

input_ids = data['input_ids']           # data contains the first batch of train data
attention_mask = data['attention_mask']

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

F.softmax(model(input_ids, attention_mask), dim=1)  # apply to the model without training



# ========== Training ==========

'''
How do we come up with all hyperparameters? The BERT authors have some recommendations for fine-tuning:
Batch size: 16, 32
Learning rate (Adam): 5e-5, 3e-5, 2e-5
Number of epochs: 2, 3, 4
We're going to ignore the number of epochs recommendation but stick with the rest. Note that increasing the batch size reduces the training time significantly, but gives you lower accuracy.
'''


EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS       # number of batches in training data * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss()


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):  # with training data
  
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"]
    attention_mask = d["attention_mask"]
    targets = d["targets"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # Clips gradients of an iterable of parameters.
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  print('finish data loader')
  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, n_examples):   # with validation data
  
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"]
      attention_mask = d["attention_mask"]
      targets = d["targets"]

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)



%%time

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, len(df_val))

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc.cpu().detach().numpy())
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc.cpu().detach().numpy())
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);


# ========== Evaluation ==========

test_acc, _ = eval_model(model, test_data_loader, loss_fn, len(df_test))

test_acc.item()

def get_predictions(model, data_loader):
  
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"]
      attention_mask = d["attention_mask"]
      targets = d["targets"]

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()   # E.g., prediction_probs.shape: (788, 3)
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader) # E.g., y_pred_probs.shape: (788, 3)

y_test

print(classification_report(y_test, y_pred)) # E.g., len(y_test): 788; len(y_pred): 788 #, target_names=class_names

# confusion matrix
def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)



idx = 2

review_text = y_review_texts[idx]     # review text in test data
true_sentiment = y_test[idx]          # true sentiment
pred_df = pd.DataFrame({
  'class_names': class_names,         # class_names = ['negative', 'neutral', 'positive']
  'values': y_pred_probs[idx]         #         E.g., [0.05, 0.9, 0.05]
})


print("\n".join(wrap(review_text)))
print()
print(f'True sentiment: {class_names[true_sentiment]}')


sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1]);


# ========== Prediction ==========

review_text = "Wait too long."

encoded_review = tokenizer.encode_plus(
  review_text,
  max_length=MAX_LEN,
  add_special_tokens=True,
  return_token_type_ids=False,
  padding='max_length',
  return_attention_mask=True,
  return_tensors='pt',
  truncation=True
)

input_ids = encoded_review['input_ids']
attention_mask = encoded_review['attention_mask']

output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)

print(f'Review text: {review_text}')
print(f'Sentiment  : {class_names[prediction]}')


df_biz = pd.read_csv("azureml://subscriptions/f49ecf56-61d3-4ccf-8ce0-431494d09f42/resourcegroups/artemis-k8s/workspaces/automl-dev/datastores/victorialystore/paths/bert/business_sample.csv", encoding='ISO-8859-1')

df_biz.head()


df_biz.shape

df_biz['Category Name '].unique()
df_biz['Prediction'] = None
df_biz['Case Id']
# drop missing data
missing_count_1 = df_biz['Verbatim'].isna().sum()
missing_count_2 = df_biz['Category Name '].isna().sum()
print("missing data: " + str(missing_count_1 + missing_count_2))
df_biz = df_biz.dropna(subset=['Verbatim'])
df_biz = df_biz[df_biz != 'N_A'].dropna(subset=['Category Name '])

# drop duplicate data
#duplicate_counts = df_biz['Case Id'].value_counts()
#print("duplicate data: " + str(duplicate_counts))
df_biz.drop_duplicates(subset=['Case Id'], inplace=True)

# change column name
df_biz = df_biz.rename(columns={'Verbatim': 'content'})
df_biz = df_biz.rename(columns={'Category Name ': 'sentiment'})

records = df_biz.shape[0]
print("number of data: " + str(records))

df_biz.head()

biz_data_loader = create_data_loader(df_biz, tokenizer, MAX_LEN, BATCH_SIZE)

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(model, biz_data_loader)

y_test

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

for index, row in df_biz.iterrows():
    review_text = row['Verbatim']
    encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    truncation=True
    )

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    if class_names[prediction] == 'class 6':
        result = 'Compliment'
    elif class_names[prediction] == 'class 5':
        result = 'Compliment'
    elif class_names[prediction] == 'class 4':
        result = 'Compliment'
    elif class_names[prediction] == 'class 3':
        result = 'Compliment'
    elif class_names[prediction] == 'class 2':
        result = 'Feedback'
    elif class_names[prediction] == 'class 1':
        result = 'Suggestion'
    elif class_names[prediction] == 'class 0':
        result = 'Complaint'

    df_biz.at[index, "Prediction"] = result
    

df_biz.head()


accurate = 0
for index, row in df_biz.iterrows():
    if row['Category Name '] == row['Prediction']:
        accurate += 1


print(f'Number of Accurate Prediction: {accurate}')
print(f'Number of Data: {len(df_biz)}')
print(f'Accuracy Rate: {accurate/len(df_biz)}')


for index, row in df_biz.iterrows():
    if row['Category Name '] != row['Prediction']:
        print(row['Verbatim'])
        print(row['Category Name '])
        print(row['Prediction'])
        print('-'*20)


