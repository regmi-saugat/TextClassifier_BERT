from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
import numpy as np
import evaluate
import config

class TextClassificationModelTrainer:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.model_checkpoint = config.MODEL_CHECKPOINT
        self.id2label = config.ID2LABEL
        self.label2id = config.LABEL2ID
        self.number_labels = len(self.id2label)
        self.device = config.DEVICE
        self.evaluation_metric = config.EVALUATION_METRIC
        self.model_output_dir = config.MODEL_OUTPUT_DIR
        self.number_epochs = config.NUMBER_EPOCHS
        self.lr = config.LR
        self.batch_size = config.BATCH_SIZE
        self.weight_decay = config.WEIGHT_DECAY
        self.evaluation_strategy = config.EVALUATION_STRATEGY
        self.save_strategy = config.SAVE_STRATEGY
        self.logging_strategy = config.LOGGING_STRATEGY
        self.push_to_hub = config.PUSH_TO_HUB
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint,
                                                                        id2label = self.id2label,
                                                                        label2id = self.label2id,
                                                                        num_labels = self.number_labels
                                                                        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.evaluation_metric_computer = evaluate.load(self.evaluation_metric)
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    def compute_metrics(self, evaluation_prediction):
        predictions, labels = evaluation_prediction
        predictions = np.argmax(predictions, axis=1)
        return self.evaluation_metric_computer.compute(predictions=predictions, references=labels)

    def set_training_arguments(self):
        return TrainingArguments(
            output_dir = self.model_output_dir,
            num_train_epochs = self.number_epochs,
            learning_rate = self.lr,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size = self.batch_size,
            weight_decay = self.weight_decay,
            evaluation_strategy = self.evaluation_strategy,
            save_strategy = self.save_strategy,
            logging_strategy = self.logging_strategy,
            push_to_hub = self.push_to_hub
        )

    def model_trainer(self):
        return Trainer(
            model = self.model,
            args = self.set_training_arguments(),
            data_collator = self.data_collator,
            train_dataset = self.train_df,
            eval_dataset = self.test_df,
            compute_metrics = self.compute_metrics
        )

    def train_and_save_and_push_to_hub(self):
        trainer = self.model_trainer()
        trainer.train()
        trainer.push_to_hub

