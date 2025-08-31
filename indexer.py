import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data with error handling
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")
    print("Please ensure you have an internet connection and try again.")
    exit(1)

def process(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
        
    text = text.lower()
    
    tokens = word_tokenize(text)
    
    processed_tokens = []
    for token in tokens:
        if token.isalpha():
            if token not in set(stopwords.words('english')):
                stemmed_token = PorterStemmer().stem(token)
                processed_tokens.append(stemmed_token)
                
    return processed_tokens

DATA_FILE = './data/publications.json'
INDEX_FILE = './data/index.pkl'

def build_index():
    print("Starting indexer for TF-IDF and Positional Index...")

    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            publications = json.load(f)
            if not publications:
                print(f"Error: No publications found in {DATA_FILE}")
                return
    except FileNotFoundError:
        print(f"Error: Crawled data file not found at {DATA_FILE}.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {DATA_FILE}")
        return
    except Exception as e:
        print(f"Error reading publications file: {str(e)}")
        return

    positional_index = {}
    doc_store = {}
    corpus = [] 

    print("Building positional index...")
    for doc_id, doc in enumerate(publications):
        try:
            # Safely get document fields with defaults
            authors = doc.get('authors', [])
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            
            # Ensure authors is a list
            if isinstance(authors, str):
                authors = [authors]
            elif not isinstance(authors, list):
                authors = []
                
            author_names = ' '.join(authors)
            content = f"{title} {author_names} {abstract}".strip()
            
            doc_store[doc_id] = doc
            corpus.append(content)

            tokens = process(content)
            
            for pos, token in enumerate(tokens):
                if token not in positional_index:
                    positional_index[token] = {}
                if doc_id not in positional_index[token]:
                    positional_index[token][doc_id] = []
                positional_index[token][doc_id].append(pos)
                
        except Exception as e:
            print(f"Warning: Skipping document {doc_id} due to error: {str(e)}")
            continue

    print("\nCreating TF-IDF model...")
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")

    index_data = {
        'positional_index': positional_index,
        'doc_store': doc_store,
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer
    }
    
    try:
        joblib.dump(index_data, INDEX_FILE)
        print(f"\nIndexing complete. Indexed {len(publications)} documents.")
        print(f"Index saved to {INDEX_FILE}")
    except Exception as e:
        print(f"Error saving index file: {str(e)}")
        return

if __name__ == '__main__':
    build_index()
