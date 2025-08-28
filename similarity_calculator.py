import json
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

with open('easy_chair_summary_of_papers_Ollama_keywords.json', 'r', encoding='utf-8') as f:
    paper_data = json.load(f)

with open('SS_dataset_summary_of_papers_Ollama_keywords.json', 'r', encoding='utf-8') as f:
    reviewer_data = json.load(f)


def extract_keywords(llm_output):
    keywords = []

   
    lines = llm_output.strip().split('\n')

   
    numbered_pattern = re.compile(r'^\s*\d+[-.)]?\s*(.*)$')  
    bullet_pattern = re.compile(r'^\s*[-*•]\s*(.*)$')

    for line in lines:
        line = line.strip()
        if not line:
            continue  

        
        num_match = numbered_pattern.match(line)
        if num_match:
            content = num_match.group(1)
            keywords_in_line = re.split(r',|;', content)
            keywords.extend([kw.strip() for kw in keywords_in_line if kw.strip()])
            continue

        bullet_match = bullet_pattern.match(line)
        if bullet_match:
            content = bullet_match.group(1)
            keywords_in_line = re.split(r',|;', content)
            keywords.extend([kw.strip() for kw in keywords_in_line if kw.strip()])
            continue

        if ':' in line:
            _, content = line.split(':', 1)
            keywords_in_line = re.split(r',|;', content)
            keywords.extend([kw.strip() for kw in keywords_in_line if kw.strip()])
            continue

        if ',' in line or ';' in line:
            keywords_in_line = re.split(r',|;', line)
            keywords.extend([kw.strip() for kw in keywords_in_line if kw.strip()])
            continue

        if len(line.split()) <= 5:
            keywords.append(line)

    return keywords


def preprocess_keywords(keywords):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_keywords = []
    for keyword in keywords:
        keyword = keyword.lower()
        keyword = re.sub(r'[^\w\s]', '', keyword)
        tokens = nltk.word_tokenize(keyword)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        if tokens:
            processed_keyword = ' '.join(tokens)
            processed_keywords.append(processed_keyword)

    return processed_keywords

paper_keywords_dict = {}
for paper_id, paper_info in tqdm(paper_data.items()):
    llm_output = paper_info[-1]
    keywords = extract_keywords(llm_output)
    processed_keywords = preprocess_keywords(keywords)
    paper_keywords_dict[paper_id] = processed_keywords


reviewer_keywords_dict = {}
for reviewer_key, papers in tqdm(reviewer_data.items()):
    reviewer_keywords = []
    for paper_info in papers:
        llm_output = paper_info[-1]
        keywords = extract_keywords(llm_output)
        processed_keywords = preprocess_keywords(keywords)
        reviewer_keywords.extend(processed_keywords)
    reviewer_keywords = list(set(reviewer_keywords))
    reviewer_keywords_dict[reviewer_key] = reviewer_keywords

paper_documents = []
paper_ids = []
for paper_id, keywords in tqdm(paper_keywords_dict.items()):
    document = ' '.join(keywords)
    paper_documents.append(document)
    paper_ids.append(paper_id)

reviewer_documents = []
reviewer_ids = []
for reviewer_id, keywords in tqdm(reviewer_keywords_dict.items()):
    document = ' '.join(keywords)
    reviewer_documents.append(document)
    reviewer_ids.append(reviewer_id)

vectorizer = TfidfVectorizer()
all_documents = paper_documents + reviewer_documents
vectorizer.fit(all_documents)
paper_tfidf = vectorizer.transform(paper_documents)
reviewer_tfidf = vectorizer.transform(reviewer_documents)

cosine_sim_matrix = cosine_similarity(paper_tfidf, reviewer_tfidf)

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    else:
        return len(intersection) / len(union)


jaccard_sim_matrix = np.zeros((len(paper_ids), len(reviewer_ids)))
for i, paper_id in tqdm(enumerate(paper_ids)):
    paper_keywords = set(paper_keywords_dict[paper_id])
    for j, reviewer_id in enumerate(reviewer_ids):
        reviewer_keywords = set(reviewer_keywords_dict[reviewer_id])
        sim = jaccard_similarity(paper_keywords, reviewer_keywords)
        jaccard_sim_matrix[i, j] = sim


cosine_sim_matrix = cosine_sim_matrix
jaccard_sim_matrix = jaccard_sim_matrix
combined_sim_matrix = (cosine_sim_matrix + jaccard_sim_matrix) / 2


matches = {}
for i, paper_id in tqdm(enumerate(paper_ids)):
    sim_scores = combined_sim_matrix[i]
    all_reviewers = [(reviewer_ids[j], sim_scores[j]) for j in range(len(reviewer_ids))]
    sorted_reviewers = sorted(all_reviewers, key=lambda x: x[1], reverse=True)
    matches[paper_id] = sorted_reviewers
with open('paper_reviewer_matches_llamma_summary_combined.json', 'w', encoding='utf-8') as f:
    json.dump(matches, f, ensure_ascii=False, indent=4)



matches = {}
for i, paper_id in tqdm(enumerate(paper_ids)):
    sim_scores = cosine_sim_matrix[i]
    all_reviewers = [(reviewer_ids[j], sim_scores[j]) for j in range(len(reviewer_ids))]
    sorted_reviewers = sorted(all_reviewers, key=lambda x: x[1], reverse=True)
    matches[paper_id] = sorted_reviewers
with open('paper_reviewer_matches_llamma_summary_cosine.json', 'w', encoding='utf-8') as f:
    json.dump(matches, f, ensure_ascii=False, indent=4)


matches = {}
for i, paper_id in tqdm(enumerate(paper_ids)):
    sim_scores = jaccard_sim_matrix[i]
    all_reviewers = [(reviewer_ids[j], sim_scores[j]) for j in range(len(reviewer_ids))]
    # Sort reviewers by similarity score in descending order
    sorted_reviewers = sorted(all_reviewers, key=lambda x: x[1], reverse=True)
    matches[paper_id] = sorted_reviewers
with open('paper_reviewer_matches_llamma_summary_jaccard.json', 'w', encoding='utf-8') as f:
    json.dump(matches, f, ensure_ascii=False, indent=4)