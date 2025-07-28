import json
import nltk
import numpy as np
import tensorflow as tf
import random
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required nltk data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = set(stopwords.words('english'))

# Tokenize and classify patterns
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize, lowercase, remove stopwords
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words and w.isalpha()]
words = sorted(set(words))
classes = sorted(set(classes))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0] if w.isalpha()]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Define model architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Compile model (FIXED: removed .legacy)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save model and metadata
model.save("chatbot_model.h5")
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
