import os
import re

def load_text_data(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                text = re.sub(r'[^\w\s]', '', text.lower())
                texts.append(text.split())  # Split into words
    return texts