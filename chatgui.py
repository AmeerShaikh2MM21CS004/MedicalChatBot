import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
import pickle
import json
import random
from tkinter import *
from tensorflow.keras.models import load_model

# Load assets
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# --- NLP Functions ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    if not ints:
        return "I'm sorry, I don't understand that."
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure how to help with that."

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

# --- GUI Actions ---
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "\U0001F464 You:\n", "user_label")
        ChatLog.insert(END, msg + '\n\n', "user_msg")

        res = chatbot_response(msg)
        ChatLog.insert(END, "\U0001F916 MedBot:\n", "bot_label")
        ChatLog.insert(END, res + '\n\n', "bot_msg")

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

# --- GUI Layout ---
base = Tk()
base.title("🩺 Medical Chatbot")
base.geometry("500x600")
base.resizable(width=False, height=False)
base.configure(bg='#f0f0f0')

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font=("Segoe UI", 12), wrap=WORD)
ChatLog.config(state=DISABLED)
ChatLog.tag_configure("user_label", foreground="blue", font=("Segoe UI", 10, "bold"))
ChatLog.tag_configure("bot_label", foreground="green", font=("Segoe UI", 10, "bold"))
ChatLog.tag_configure("user_msg", lmargin1=10, foreground="black")
ChatLog.tag_configure("bot_msg", lmargin1=10, foreground="black")

scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

entry_frame = Frame(base, bg="#f0f0f0")
EntryBox = Text(entry_frame, bd=0, bg="white", width="36", height="3", font=("Segoe UI", 12))
EntryBox.bind("<Return>", lambda event: send())

SendButton = Button(entry_frame, font=("Segoe UI", 12, 'bold'), text="Send", width=12, height=2, bd=0,
                    bg="#007acc", activebackground="#005f99", fg='white', command=send)

# Placement
scrollbar.place(x=470, y=6, height=486)
ChatLog.place(x=6, y=6, height=486, width=460)
entry_frame.place(x=6, y=500, width=488, height=90)
EntryBox.pack(side=LEFT, padx=(5, 5), pady=5)
SendButton.pack(side=RIGHT, padx=(5, 5), pady=5)

base.mainloop()
