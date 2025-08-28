import json
import ollama
from tqdm import tqdm

with open('SS_dataset_with_summaries.json', 'r') as f:
    data = json.load(f)

def get_keywords(summary):
    prompt = f"Generate a set of keywords that would help find a reviewer for the following summary of the paper abstract: '{summary}'"
    response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
    keywords = response['message']['content']
    return keywords.strip()

for author_key, papers in tqdm(data.items()):
    for paper in papers:
        summary = paper[6]
        keywords = get_keywords(summary)
        paper.append(keywords)

with open('SS_dataset_summary_of_papers_LLAMA_keywords.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)

print("Keywords have been successfully generated and added to each paper entry.")
