

def train():
    import numpy as np 
    import pandas as pd
    import re
    import string
    import os

    import logging
    import boto3
    from botocore.exceptions import ClientError

    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding,Bidirectional, GlobalMaxPool1D, SpatialDropout1D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
    from tensorflow.keras.metrics import Precision, Recall
    from sklearn import metrics
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset

    # load data from s3
    s3_client = boto3.client('s3')
    dataset_bucket_name = os.environ.get('DATA_S3_BUCKET_NAME')
    file_key = 'data.csv'
    s3_client.download_file(dataset_bucket_name, file_key, 'data.csv')
    df=pd.read_csv('data.csv')

    # train test split
    TEST_SPLIT = 0.2
    RANDOM_STATE = 10
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label_text"],test_size = TEST_SPLIT, random_state = RANDOM_STATE)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    texts_train=list(X_train)
    labels_train=list(y_train)
    texts_test=list(X_test)
    labels_test=list(y_test)

    """# Pre process steps

    1. Stemming and Lemmatization
    2. Tokenizer
    3. text to sequence
    4. pad_sequence
    5. one hot encoding

    ##### Stemming and Lemmatization
    """

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    patterns = []
    tags = []
    for i in range(len(texts_train)):
        # Convert all text to lowercase
        pattern = texts_train[i].lower()
            
        # Remove non-alphanumeric characters and replace them with space
        pattern = re.sub(r'[^a-z0-9]', ' ', pattern)
            
        # Tokenize text
        tokens = nltk.word_tokenize(pattern)
            
        # Remove stop words
        tokens = [token for token in tokens if token not in stop_words]
            
        # Apply lemmatization and stemming
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [stemmer.stem(token) for token in tokens]
            
        # Join the tokens back into a string
        pattern = ' '.join(tokens)
            
        # Append the pattern and tag to respective lists
        patterns.append(pattern)
        tags.append(labels_train[i])

    """##### Tokenizer"""

    unique_words = set()
    for text in texts_train:
        words = nltk.word_tokenize(text.lower())
        unique_words.update(words)
    len(unique_words)
    unique_word_len=len(unique_words)
    num_words=unique_word_len+100
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(patterns)

    """##### Text to Sequence"""

    max_sequence_len = max([len(tokenizer.texts_to_sequences(patterns)[i]) for i in range(len(patterns))])
    sequences = tokenizer.texts_to_sequences(patterns)
    max_sequence_len=max_sequence_len+100

    """##### Pad Sequences"""

    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

    """##### One Hot encoding"""

    training = np.array(padded_sequences)
    output = np.array(tags)
    output_labels = np.unique(output)
    encoder = LabelEncoder()
    encoder.fit(output)
    encoded_y = encoder.transform(output)
    output_encoded = tf.keras.utils.to_categorical(encoded_y)

    """# Create Model"""

    VAL_SPLIT = 0.1
    BATCH_SIZE = 10
    EPOCHS = 20
    EMBEDDING_DIM = 32
    NUM_UNITS = 32
    NUM_CLASSES=len(set(labels_train))
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(input_dim = VOCAB_SIZE, output_dim = EMBEDDING_DIM, input_length = max_sequence_len, mask_zero = True),
        Dropout(0.2),
        LSTM(NUM_UNITS,activation='relu'),
        Dense(len(output_labels), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Precision(), Recall(),'accuracy'])
    print(model.summary())

    """# Train model"""
    history=model.fit(training, output_encoded, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose = 1, validation_split = VAL_SPLIT)

    """# Save The Tokenizer"""

    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    """# Save Model"""

    from mlem.api import save,load
    save(model, "models/tf")
    save(encoder,"encoder/tf")    
    
    file_name='tokenizer.pickle'
    bucket_name = os.environ.get('MODEL_S3_BUCKET_NAME')
    try:
        response = s3_client.upload_file(file_name, bucket_name, 'tokenizer.pickle')
        folder_path = 'models'
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            response=s3_client.upload_file(file_path, bucket_name, f'models/{file_name}')
        folder_path = 'encoder'
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            response=s3_client.upload_file(file_path, bucket_name, f'encoder/{file_name}')
    except ClientError as e:
        logging.error(e)
        return False
    return True

if __name__ == '__main__':
    result = train()
    print(result)

