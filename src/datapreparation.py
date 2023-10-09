#@ IMPORTING THE REQUIRED LIBRARIES AND DEPENDENCIES
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
import config
import torch


class TextClassificationDataset:

    def __init__(self):
        self.dataset_id = config.DATASET_ID
        self.model_checkpoint = config.MODEL_CHECKPOINT
        self.source_column = config.SOURCE_COLUMN
        self.target_column = config.TARGET_COLUMN
        self.test_size = config.TEST_SIZE
        self.seed = config.SEED
        self.max_len = config.MAX_LENGTH
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

    def create_data(self):
        self.dataset = load_dataset(self.dataset_id, split="train")
        self.data = self.dataset.to_pandas()
        self.data = self.data[[self.source_column, self.target_column]]
        self.data[self.target_column] = self.data[self.target_column].apply(lambda x: 2 if x == -1 else x)
        self.data[self.source_column] =  self.data[self.source_column].apply(lambda x: x.lower())                       # lowercasing the dataset
        self.data = self.data.sample(20000)
        self.train_data, self.test_data = train_test_split(self.data, test_size=self.test_size, shuffle=True, random_state=self.seed, stratify=self.data[self.target_column])
        self.train_df = Dataset.from_pandas(self.train_data)
        self.test_df = Dataset.from_pandas(self.test_data)
        return self.train_df, self.test_df


    def tokenize_function(self, example):
        model_input = self.tokenizer(example[self.source_column], truncation=True, padding=True, max_length=self.max_len)
        labels = torch.tensor(example[self.target_column], dtype=torch.int)
        model_input["labels"] = labels
        return model_input

    def preprocess_function(self, data):
        model_input = data.map(self.tokenize_function, batched=True, remove_columns=data.column_names)
        return model_input

    def gen_classification_dataset(self):
        train_df, test_df = self.create_data()
        train_tokenized_data = self.preprocess_function(train_df)
        test_tokenized_data = self.preprocess_function(test_df)
        return train_tokenized_data, test_tokenized_data
