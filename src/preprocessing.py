import numpy as np
import utils as u

# Load raw tweets
tweets_train_pos = u.txt_to_tweets_train(u.PATH_TRAIN_POS)
tweets_train_neg = u.txt_to_tweets_train(u.PATH_TRAIN_NEG)

# Cleanup tweets
tweets_train_pos = u.cleanup(tweets_train_pos, u.PATH_CLEAN_POS)
tweets_train_neg = u.cleanup(tweets_train_neg, u.PATH_CLEAN_NEG)

# Load clean tweets
tweets_train_pos = u.txt_to_tweets_train(u.PATH_CLEAN_POS)
tweets_train_neg = u.txt_to_tweets_train(u.PATH_CLEAN_NEG)

# Create training and validation sets
u.create_training_set(tweets_train_pos, tweets_train_neg)