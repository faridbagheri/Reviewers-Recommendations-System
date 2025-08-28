import json
import ollama
from tqdm import tqdm

with open('EC_dizionario_articoli.json', 'r') as f:
    data = json.load(f)

def get_keywords(title):
    prompt = f"Generate a set of keywords that would help find a reviewer for the following paper title: '{title}'"
    response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
    keywords = response['message']['content']
    return keywords.strip()

for key, values in tqdm(data.items()):
    title = values[1]
    keywords = get_keywords(title)
    data[key].append(keywords)

with open('easy_chair_title_of_papers_LLAMA_keywords.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

print("Keywords have been successfully generated and added to each paper entry.")
