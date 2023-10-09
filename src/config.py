import torch

class Config:
    DATASET_ID = "emad12/stock_tweets_sentiment"
    MODEL_CHECKPOINT = "distilbert-base-uncased"
    SOURCE_COLUMN = "tweet"
    TARGET_COLUMN = "sentiment"
    TEST_SIZE = 0.2
    SEED = 0
    MAX_LENGTH = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ID2LABEL = {0: "NEUTRAL", 1: "POSITIVE", 2: "NEGATIVE"}
    LABEL2ID = {"NEUTRAL" : 0, "POSITIVE" : 1, "NEGATIVE": 2}
    EVALUATION_METRIC = "accuracy"
    MODEL_OUTPUT_DIR = "distilbert-stock-tweet-sentiment-analysis"
    NUMBER_EPOCHS = 3
    LR = 2E-5
    BATCH_SIZE = 16
    WEIGHT_DECAY = 0.01
    EVALUATION_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOGGING_STRATEGY = "epoch"
    PUSH_TO_HUB = True

config = Config()
