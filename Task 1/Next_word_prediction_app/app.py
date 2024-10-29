import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import re
import warnings

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
torch._dynamo.config.suppress_errors = True

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Supported Device: {device}\n\n")
file = open("1661-0.txt", "r")
Sherlock = file.read()
filtered_text = re.sub(r"[^a-zA-Z0-9\s.]", "", Sherlock)  # Keep alphanumeric and period
words = (filtered_text.lower().split())[0:int((107510)/2.5)]  # Split by whitespace and lowercase
# print(len(words))

# MLP Model Definition
class CustomMLPForNextWord(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_length, activation_function):
        super(CustomMLPForNextWord, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.embedding(x)  # Get embeddings
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model from file
def load_model(vocab_size, embedding_dim, hidden_size, context_length, activation):
    model = CustomMLPForNextWord(vocab_size, embedding_dim, hidden_size, context_length, torch.relu).to(device)
    model_filename = f"model_state_emb{embedding_dim}_ctx{context_length}_{activation}.pth"
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    return model







unique_words = (sorted(set(words)))
unique_words = unique_words[:-1]
unique_words =  unique_words + [" "]

word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
vocab_size = len(unique_words)
# print(vocab_size)




# def predict_next_words(starting_seq: str, model,context_length, prediction_length: int):
#     model.eval()
#     generated_text = starting_seq

#     # Tokenize and encode the starting sequence
#     starting_seq = re.sub(r"[^a-zA-Z0-9\s.]", "", starting_seq.lower()).split()
#     starting_seq = starting_seq[-context_length:] if len(starting_seq) > context_length else starting_seq
#     input_indices = [word_to_idx.get(word, 0) for word in starting_seq]

#     for _ in range(prediction_length):
#         with torch.no_grad():
#             input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
#             output = model(input_tensor)
#             pred_idx = output.argmax(dim=1).item()
#             next_word = idx_to_word[pred_idx]

#             generated_text += " " + next_word
#             input_indices = input_indices[1:] + [pred_idx]  # Update context window with predicted word

#     return generated_text

def predict_next_words(starting_seq: str, model,context_length, prediction_length: int):
    model.eval()
    generated_text = starting_seq

    # Tokenize and encode the starting sequence
    starting_seq = re.sub(r"[^a-zA-Z0-9\s.]", "", starting_seq.lower()).split()
    if len(starting_seq) > context_length :
      starting_seq = starting_seq[-context_length:] 
    elif len(starting_seq) < context_length:
        starting_seq = [""] * (context_length - len(starting_seq)) + starting_seq
    else : 
      starting_seq = starting_seq
    input_indices = [word_to_idx.get(word, 0) for word in starting_seq]

    for _ in range(prediction_length):
        with torch.no_grad():
            input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            next_word = idx_to_word[pred_idx]

            generated_text += " " + next_word
            input_indices = input_indices[1:] + [pred_idx]  # Update context window with predicted word

    return generated_text





def main():
    st.title("Next Word Prediction on Sherlock Holmes Dataset")


    st.sidebar.title("Model Parameters")

# Sidebar sliders
    hidden_size = 1024
    context_length = st.sidebar.selectbox(
    "Select Context length. ", (3, 5, 10))

    activation = st.sidebar.selectbox(
    "Select Activation Function. ", ("tanh", "ReLU"))

    embedding_dim = st.sidebar.selectbox(
    "Select embedding dimension. ", (32, 64, 128))
    k = st.sidebar.slider("Select the number of words to predict.", min_value=0, max_value=3000, value=50, step=1)

 

    st.write("Hidden Layer Size:", hidden_size, "Selected Context Length:", context_length, "Embedding Dimension:", embedding_dim, "Activation Function:", activation )
  




    starting_sequence = st.text_input("Enter Starting Text")
    st.write("Starting Sequence:", starting_sequence)


    # Load model and check if it exists
    model = load_model(vocab_size, embedding_dim, hidden_size, context_length, activation)
    if model is None:
        st.error("Failed to load model.")
        return

    # Predict next words
    if st.button("Predict Next Words"):
        if starting_sequence:
            generated_text = predict_next_words(starting_sequence, model, context_length, k)
            st.write("Generated Text:", generated_text)
        else:
            st.warning("Please enter a starting text sequence.")

if __name__ == "__main__":
    main()
