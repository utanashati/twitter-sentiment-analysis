import numpy as np

PATH_TRAIN_POS = "data/train_pos.txt"
PATH_TRAIN_POS_FULL = "data/train_pos_full.txt"
PATH_TRAIN_NEG = "data/train_neg.txt"
PATH_TRAIN_NEG_FULL = "data/train_neg_full.txt"
PATH_TEST = "data/test_data.txt"
PATH_CLEAN_POS = 'data/train_pos_clean.txt'
PATH_CLEAN_NEG = 'data/train_neg_clean.txt'
PATH_CLEAN_POS_FULL = 'data/train_pos_clean_full.txt'
PATH_CLEAN_NEG_FULL = 'data/train_neg_clean_full.txt'
PATH_TRAIN = 'data/train.txt'
PATH_VALID = 'data/valid.txt'
PATH_TRAIN_FULL = 'data/train_full.txt'
PATH_VALID_FULL = 'data/valid_full.txt'
PATH_TRAIN_FULL_1 = 'data/train_full_1.txt'
PATH_TRAIN_FULL_2 = 'data/train_full_2.txt'
PATH_TRAIN_FULL_3 = 'data/train_full_3.txt'
PATH_TRAIN_FULL_4 = 'data/train_full_4.txt'

PATHS_TRAIN = (PATH_TRAIN_FULL_1, PATH_TRAIN_FULL_2, PATH_TRAIN_FULL_3, PATH_TRAIN_FULL_4)

ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789' \
            '-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
DICT_CHAR_TO_IND = {char: i for i, char in enumerate(ALPHABET)}
DICT_IND_TO_CHAR = {i: char for i, char in enumerate(ALPHABET)}

MAX_WORDS_TWEET = 35
MAX_CHARS_WORD = 16
BATCHSIZE = 32

def get_train_steps(paths=PATHS_TRAIN, batchsize=BATCHSIZE):
	counter = 0
	for path in paths:
		with open(path, 'r', encoding='latin-1') as f:
			for line in f:
				counter += 1

	return counter // batchsize

def get_valid_steps(path=PATH_VALID_FULL, batchsize=BATCHSIZE):
	counter = 0
	with open(path, 'r', encoding='latin-1') as f:
		for line in f:
			counter += 1

	return counter // batchsize

def txt_to_tweets_train(path):
    tweets = []
    with open(path, 'r', encoding='latin-1') as f:
        for tweet in f:
        	tweets.append(tweet[:-1]) # don't read last '\n'
            
    return tweets

def txt_to_tweets_test(path=PATH_TEST):
    tweets = []
    with open(path, 'r', encoding='latin-1') as f:
        for tweet in f:
            tweet = tweet[:-1]
            tweets.append(tweet.split(',', 1))
            
    return tweets

def print_all(np_array):
	np.set_printoptions(threshold=np.nan)
	print(np_array)
	np.set_printoptions()

def cleanup(tweets, path=None):
	tweets = list(map(lambda t: t.replace(u'\u201c', '"').replace(u'\u201d', '"'), tweets)) # remove curly quotes
	tweets = list(map(lambda t: t.replace('<user> ', ''), tweets)) # remove <user>
	tweets = list(map(lambda t: t.replace('<url> ', '').strip(), tweets)) # remove <url>
	tweets = list(set(tweets)) # remove duplicates
	tweets = list(filter(None, tweets)) # delete empty tweets
	if path:
	    with open(path, 'w') as f:
	        for tweet in tweets:
	            f.write(tweet + '\n')

	return tweets	           

def create_training_set(tweets_pos, tweets_neg, valid_part=0.05, path_train=PATH_TRAIN_FULL, path_valid=PATH_VALID_FULL):
	concat = lambda x, y: np.concatenate((np.ones(len(x), dtype=int)[:, np.newaxis]*y, np.array(x)[:, np.newaxis]), axis=1)
	train = np.concatenate((concat(tweets_pos, 1), concat(tweets_neg, 0)), axis=0)
	np.random.seed(42)
	np.random.shuffle(train)
	cut_point = int(len(train)*(1-valid_part)) - 1
	np.savetxt(path_train, train[:cut_point], fmt='%s', delimiter='--')
	np.savetxt(path_valid, train[cut_point:, :], fmt='%s', delimiter='--')

def tokenize(tweets):
	for t, tweet in enumerate(tweets):
		tweets[t] = tweet.split(' ')

	return tweets

def get_max_lengths(tweets_tokd):
	max_words_tweet = 0
	max_chars_word = 0
	for tweet in tweets_tokd:
		if len(tweet) > max_words_tweet:
			max_words_tweet = len(tweet)
		for word in tweet:
			if len(word) > max_chars_word:
				max_chars_word = len(word)

	return max_words_tweet, max_chars_word

def get_lengths_distrs(tweets_tokd, max_lengths):
	max_words_tweet, max_chars_word = max_lengths
	words_tweet = [0] * max_words_tweet
	chars_word = [0] * max_chars_word
	for tweet in tweets_tokd:
		words_tweet[len(tweet)-1] += 1
		for word in tweet:
			chars_word[len(word)-1] += 1

	return words_tweet, chars_word

def cut_lengths(tweets_tokd, max_words_tweet=MAX_WORDS_TWEET, max_chars_word=MAX_CHARS_WORD):
	for t, tweet in enumerate(tweets_tokd):
		words_to_delete = []
		for w, word in enumerate(tweet):
			if len(word) > MAX_CHARS_WORD:
				words_to_delete.append(w)
		tmp = tweets_tokd[t]
		del tweets_tokd[t]
		tweets_tokd.insert(t, list(np.delete(tmp, words_to_delete, None)))

	for t, tweet in enumerate(tweets_tokd):
		if len(tweet) > MAX_WORDS_TWEET:
			tweets_tokd[t] = tweet[:MAX_WORDS_TWEET]

	return tweets_tokd

def tweet_to_onehot(tweet_tokd):
	onehot = np.zeros((MAX_WORDS_TWEET, MAX_CHARS_WORD, len(ALPHABET)))
	for w, word in enumerate(tweet_tokd):
		for c, char in enumerate(word):
			if char not in DICT_CHAR_TO_IND:
				continue
			else:
				onehot[w, c, DICT_CHAR_TO_IND[char]] = 1

	return onehot

def tweets_to_batch(tweets_tokd_batch, batchsize=BATCHSIZE):
	assert len(tweets_tokd_batch) == batchsize
	batch = np.zeros((batchsize, MAX_WORDS_TWEET, MAX_CHARS_WORD, len(ALPHABET)))
	for i in range(batchsize):
		batch[i] = tweet_to_onehot(tweets_tokd_batch[i])

	return batch

def split_txt(path, n_chunks):
	counter = 0
	with open(path, 'r', encoding='latin-1') as f:
		for line in f:
			counter += 1

	with open(path, 'r', encoding='latin-1') as f:
		chunk_size = counter // n_chunks
		for i in range(n_chunks):
			if i == n_chunks - 1:
				remaining = -1
			else:
				remaining = chunk_size
			with open(path[:-4] + '_{}.txt'.format(i+1), 'w', encoding='latin-1') as chunk:
				for line in f:
					chunk.write(line)
					remaining -= 1
					if remaining == 0:
						break

def iterate_batches(paths=PATHS_TRAIN, batchsize=BATCHSIZE):
	inputs, targets = [], []
	remaining = batchsize
	for path in paths:
		with open(path, 'r', encoding='latin-1') as t:
			for tweet in t:
				target, input_ = tweet.split('--', 1)
				input_ = input_[:-1]
				inputs.append(input_)
				targets.append(int(target))
				remaining -= 1
				if remaining == 0:
					inputs = cut_lengths(tokenize(inputs))
					inputs = tweets_to_batch(inputs, batchsize)
					yield inputs, np.array(targets)
					inputs, targets = [], []
					remaining = batchsize
	
	
