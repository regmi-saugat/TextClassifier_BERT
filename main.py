from src.datapreparation import TextClassificationDataset
from src.modeltrainer import TextClassificationModelTrainer


if __name__ == "__main__":
    textclassificationdataset = TextClassificationDataset()
    train_df, test_df = textclassificationdataset.gen_classification_dataset()
    textclassificationtrainer = TextClassificationModelTrainer(train_df, test_df)
    textclassificationtrainer.train_and_save_and_push_to_hub()
