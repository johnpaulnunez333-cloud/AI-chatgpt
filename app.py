import re
import math
import json
import random
import string
import requests
from collections import deque, Counter
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
GEMINI_API_KEY = "AIzaSyB-NGIMekeWFwtPacoEM3UR_wM9qwJEnoc"
PH_TZ = timezone(timedelta(hours=8))

class Tokenizer:
    def __init__(self):
        self.stopwords = set([
            "a","an","the","is","it","in","on","at","to","for","of","and",
            "or","but","are","was","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may",
            "might","shall","can","need","dare","used","ought","i","you",
            "he","she","we","they","me","him","her","us","them","my","your",
            "his","its","our","their","this","that","these","those","what",
            "which","who","how","when","where","why","not","no","yes","so",
        ])

    def normalize(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        return self.normalize(text).split()

    def tokenize_no_stop(self, text):
        return [w for w in self.tokenize(text) if w not in self.stopwords]

    def get_ngrams(self, text, n=2, analyzer="word"):
        if analyzer == "char_wb":
            text = f" {text} "
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        words = self.tokenize(text)
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def get_all_ngrams(self, text, ngram_range=(1,2), analyzer="word"):
        result = []
        for n in range(ngram_range[0], ngram_range[1]+1):
            result += self.get_ngrams(text, n, analyzer)
        return result
tokenizer = Tokenizer()

class TfidfVectorizer:
    def __init__(self, ngram_range=(1,2), analyzer="word", max_features=6000):
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_features = max_features
        self.vocab = {}
        self.idf_vals = {}
        self.size = 0

    def _get_grams(self, text):
        return tokenizer.get_all_ngrams(text, self.ngram_range, self.analyzer)

    def fit(self, docs):
        df = {}
        freq = Counter()
        for doc in docs:
            seen = set()
            for g in self._get_grams(doc):
                freq[g] += 1
                if g not in seen:
                    df[g] = df.get(g, 0) + 1
                    seen.add(g)
        top = [t for t, _ in freq.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top)}
        self.size = len(self.vocab)
        N = len(docs)
        self.idf_vals = {
            t: math.log((N + 1) / (df.get(t, 0) + 1)) + 1
            for t in self.vocab
        }
        return self

    def transform(self, docs):
        result = []
        for doc in docs:
            vec = [0.0] * self.size
            grams = self._get_grams(doc)
            tf = Counter(grams)
            total = max(len(grams), 1)
            for g, cnt in tf.items():
                if g in self.vocab:
                    vec[self.vocab[g]] = (cnt / total) * self.idf_vals.get(g, 1.0)
            result.append(vec)
        return result

    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x*x for x in a))
    mb = math.sqrt(sum(y*y for y in b))
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)
def top_cosine(query_vec, matrix):
    sims = [cosine_sim(query_vec, row) for row in matrix]
    best = max(range(len(sims)), key=lambda i: sims[i])
    return best, sims[best]

class NaiveBayes:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.classes_ = []
        self.log_prior = {}
        self.log_likelihood = {}
        self.n_features = 0

    def fit(self, X, y):
        self.classes_ = list(set(y))
        self.n_features = len(X[0])
        counts = Counter(y)
        total = len(y)
        self.log_prior = {c: math.log(counts[c] / total) for c in self.classes_}
        feat_sum = {c: [0.0]*self.n_features for c in self.classes_}
        for xi, yi in zip(X, y):
            for j, v in enumerate(xi):
                feat_sum[yi][j] += v
        self.log_likelihood = {}
        for c in self.classes_:
            denom = sum(feat_sum[c]) + self.alpha * self.n_features
            self.log_likelihood[c] = [
                math.log((feat_sum[c][j] + self.alpha) / denom)
                for j in range(self.n_features)
            ]
        return self

    def predict_proba(self, X):
        out = []
        for xi in X:
            scores = {}
            for c in self.classes_:
                s = self.log_prior[c]
                for j, v in enumerate(xi):
                    if v > 0:
                        s += v * self.log_likelihood[c][j]
                scores[c] = s
            mx = max(scores.values())
            exp_s = {c: math.exp(s - mx) for c, s in scores.items()}
            tot = sum(exp_s.values())
            out.append({c: exp_s[c]/tot for c in self.classes_})
        return out

    def predict(self, X):
        return [max(p, key=p.get) for p in self.predict_proba(X)]

class LogisticRegression:
    def __init__(self, lr=0.05, max_iter=800, reg=0.001):
        self.lr = lr
        self.max_iter = max_iter
        self.reg = reg
        self.classes_ = []
        self.W = {}
        self.b = {}

    def _softmax(self, scores):
        mx = max(scores.values())
        exp_s = {c: math.exp(s - mx) for c, s in scores.items()}
        tot = sum(exp_s.values())
        return {c: exp_s[c]/tot for c in self.classes_}

    def fit(self, X, y):
        self.classes_ = list(set(y))
        n = len(X[0])
        self.W = {c: [0.0]*n for c in self.classes_}
        self.b = {c: 0.0 for c in self.classes_}
        for iteration in range(self.max_iter):
            lr = self.lr / (1 + 0.001 * iteration)
            for xi, yi in zip(X, y):
                raw = {c: sum(self.W[c][j]*xi[j] for j in range(n)) + self.b[c]
                       for c in self.classes_}
                probs = self._softmax(raw)
                for c in self.classes_:
                    err = probs[c] - (1.0 if c == yi else 0.0)
                    for j in range(n):
                        self.W[c][j] -= lr * (err * xi[j] + self.reg * self.W[c][j])
                    self.b[c] -= lr * err
        return self

    def predict_proba(self, X):
        n = len(X[0])
        out = []
        for xi in X:
            raw = {c: sum(self.W[c][j]*xi[j] for j in range(n)) + self.b[c]
                   for c in self.classes_}
            out.append(self._softmax(raw))
        return out

    def predict(self, X):
        return [max(p, key=p.get) for p in self.predict_proba(X)]

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, a, b):
        return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

    def predict(self, X):
        results = []
        for xi in X:
            dists = [(self._distance(xi, self.X_train[i]), self.y_train[i])
                     for i in range(len(self.X_train))]
            dists.sort(key=lambda x: x[0])
            neighbors = [label for _, label in dists[:self.k]]
            results.append(Counter(neighbors).most_common(1)[0][0])
        return results

class SentimentAnalyzer:
    def __init__(self):
        self.lexicon = {}
        positive_words = [
            ("good",3),("great",4),("awesome",5),("excellent",5),("amazing",5),
            ("wonderful",4),("fantastic",5),("happy",3),("love",4),("like",2),
            ("best",4),("nice",3),("cool",3),("perfect",5),("brilliant",4),
            ("superb",5),("outstanding",5),("enjoy",3),("impressed",4),
            ("helpful",3),("thanks",2),("thank",2),("beautiful",4),
            ("incredible",5),("positive",3),("smart",3),("fast",3),("easy",2),
            ("friendly",3),("fun",3),("interesting",3),("useful",3),
            ("powerful",4),("clear",2),("clever",4),("solved",3),("working",3),
            ("correct",3),("right",2),("genius",5),("expert",3),("galing",4),
            ("maganda",4),("masaya",4),("mabuti",3),("magaling",4),("ayos",3),
            ("wow",4),("yes",2),("agree",2),("sure",2),("okay",1),("ok",1),
        ]
        negative_words = [
            ("bad",-3),("terrible",-5),("awful",-5),("horrible",-5),("hate",-4),
            ("worst",-5),("ugly",-3),("stupid",-4),("useless",-4),("broken",-3),
            ("error",-2),("bug",-2),("problem",-2),("issue",-2),("fail",-3),
            ("failed",-3),("wrong",-3),("incorrect",-3),("disappointed",-4),
            ("boring",-2),("frustrating",-4),("annoying",-3),("slow",-2),
            ("crash",-3),("confused",-2),("difficult",-2),("hard",-2),
            ("impossible",-4),("unclear",-2),("laggy",-2),("freeze",-3),
            ("stuck",-2),("lost",-2),("dead",-3),("expired",-2),("corrupt",-3),
            ("palpak",-4),("bobo",-5),("pangit",-3),("mali",-3),("wala",-2),
        ]
        self.negators = {"not","no","never","cant","cannot","wont","dont","isnt","arent","hardly","barely"}
        self.intensifiers = {"very":1.5,"really":1.5,"so":1.3,"extremely":2.0,"absolutely":2.0,"totally":1.4}
        for word, score in positive_words:
            self.lexicon[word] = score
        for word, score in negative_words:
            self.lexicon[word] = score

    def analyze(self, text):
        words = tokenizer.tokenize(text)
        total_score = 0.0
        negate = False
        intensify = 1.0
        for i, w in enumerate(words):
            if w in self.negators:
                negate = True
                continue
            if w in self.intensifiers:
                intensify = self.intensifiers[w]
                continue
            if w in self.lexicon:
                val = self.lexicon[w] * intensify
                total_score += -val if negate else val
            negate = False
            intensify = 1.0

        normalized = 1.0 / (1.0 + math.exp(-total_score * 0.3))
        if normalized > 0.60:
            label = "positive"
        elif normalized < 0.40:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": round(normalized, 3)}

class NamedEntityRecognizer:
    def __init__(self):
        self.patterns = {
            "LANGUAGE": ["python","javascript","java","c++","php","ruby","swift","kotlin","go","rust","typescript","html","css","sql"],
            "FRAMEWORK": ["flask","django","react","angular","vue","spring","laravel","express","fastapi","tensorflow","pytorch"],
            "CONCEPT":   ["machine learning","deep learning","neural network","artificial intelligence","natural language processing","computer vision","data science","big data","cloud computing","blockchain","cybersecurity","algorithm"],
            "SCIENCE":   ["physics","chemistry","biology","mathematics","astronomy","geology","psychology","sociology","economics"],
            "PERSON":    [],
        }

    def recognize(self, text):
        text_lower = text.lower()
        found = {}
        for etype, terms in self.patterns.items():
            for term in terms:
                if term in text_lower:
                    found.setdefault(etype, []).append(term)
        return found

class TopicModeler:
    def __init__(self):
        self.topics = {
            "Technology":    ["python","code","program","software","app","web","api","html","css","javascript","flask","database","server","computer","tech","laptop","android","ios","linux","windows","git","docker"],
            "AI & ML":       ["ai","machine learning","neural","model","algorithm","dataset","training","chatbot","robot","artificial","intelligence","deep learning","nlp","gpt","transformer","prediction","classification"],
            "Science":       ["science","physics","chemistry","biology","math","mathematics","formula","equation","theory","experiment","atom","molecule","gravity","evolution","quantum","relativity","energy","force"],
            "History":       ["history","historical","ancient","war","century","civilization","empire","revolution","president","king","queen","battle","dynasty","colony","medieval","renaissance","world war"],
            "Health":        ["health","medicine","doctor","hospital","disease","symptom","treatment","diet","exercise","mental","body","pain","virus","vaccine","nutrition","fitness","sleep","stress"],
            "Education":     ["school","study","learn","education","teacher","student","university","college","exam","lesson","homework","degree","course","subject","scholarship","knowledge","research"],
            "Entertainment": ["movie","music","game","sports","play","watch","listen","read","book","film","series","anime","song","concert","tv","show","stream","netflix","gaming","youtube"],
            "Food":          ["food","eat","cook","recipe","restaurant","meal","drink","coffee","rice","bread","meat","vegetable","fruit","snack","taste","delicious","hungry","dessert"],
            "Travel":        ["travel","trip","tour","hotel","flight","vacation","country","city","beach","mountain","explore","visit","passport","airline","destination","adventure","map"],
            "Business":      ["business","money","finance","economy","market","investment","company","startup","entrepreneur","profit","salary","tax","bank","stock","crypto","trade"],
        }

    def get_topics(self, text):
        t = text.lower()
        found = [topic for topic, kws in self.topics.items() if any(k in t for k in kws)]
        return found if found else ["General"]

class MathSolver:
    def solve(self, text):
        t = text.lower()
        patterns = [
            (r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)} + {m.group(2)} = {float(m.group(1)) + float(m.group(2))}"),
            (r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)} - {m.group(2)} = {float(m.group(1)) - float(m.group(2))}"),
            (r"(\d+\.?\d*)\s*[x\*×]\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)} × {m.group(2)} = {float(m.group(1)) * float(m.group(2))}"),
            (r"(\d+\.?\d*)\s*[/÷]\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)} ÷ {m.group(2)} = {float(m.group(1)) / float(m.group(2))}"
             if float(m.group(2)) != 0 else "Cannot divide by zero!"),
            (r"(\d+\.?\d*)%\s*of\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)}% of {m.group(2)} = {float(m.group(1)) * float(m.group(2)) / 100}"),
            (r"square root of (\d+\.?\d*)",
             lambda m: f"√{m.group(1)} = {round(math.sqrt(float(m.group(1))), 6)}"),
            (r"sqrt\s*\(?(\d+\.?\d*)\)?",
             lambda m: f"√{m.group(1)} = {round(math.sqrt(float(m.group(1))), 6)}"),
            (r"(\d+\.?\d*)\s*\^\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)}^{m.group(2)} = {float(m.group(1)) ** float(m.group(2))}"),
            (r"(\d+\.?\d*)\s*to the power of\s*(\d+\.?\d*)",
             lambda m: f"{m.group(1)}^{m.group(2)} = {float(m.group(1)) ** float(m.group(2))}"),
            (r"log\s*\(?(\d+\.?\d*)\)?",
             lambda m: f"log({m.group(1)}) = {round(math.log10(float(m.group(1))), 6)}"
             if float(m.group(1)) > 0 else "log(0) is undefined"),
            (r"ln\s*\(?(\d+\.?\d*)\)?",
             lambda m: f"ln({m.group(1)}) = {round(math.log(float(m.group(1))), 6)}"
             if float(m.group(1)) > 0 else "ln(0) is undefined"),
            (r"sin\s*\(?(\d+\.?\d*)\)?",
             lambda m: f"sin({m.group(1)}°) = {round(math.sin(math.radians(float(m.group(1)))), 6)}"),
            (r"cos\s*\(?(\d+\.?\d*)\)?",
             lambda m: f"cos({m.group(1)}°) = {round(math.cos(math.radians(float(m.group(1)))), 6)}"),
            (r"tan\s*\(?(\d+\.?\d*)\)?",
             lambda m: f"tan({m.group(1)}°) = {round(math.tan(math.radians(float(m.group(1)))), 6)}"),
            (r"area of circle.*radius\s*(\d+\.?\d*)",
             lambda m: f"Area = π × {m.group(1)}² = {round(math.pi * float(m.group(1))**2, 4)}"),
            (r"circumference.*radius\s*(\d+\.?\d*)",
             lambda m: f"Circumference = 2π × {m.group(1)} = {round(2 * math.pi * float(m.group(1)), 4)}"),
            (r"(\d+\.?\d*)\s*factorial|factorial of\s*(\d+)",
             lambda m: f"{m.group(1) or m.group(2)}! = {math.factorial(int(m.group(1) or m.group(2)))}"),
        ]
        for pattern, solver in patterns:
            m = re.search(pattern, t)
            if m:
                try:
                    return "Math Result: " + solver(m)
                except Exception as e:
                    return f"Math error: {str(e)}"
        return None

class LanguageDetector:
    def __init__(self):
        self.fil_markers = set([
            "ang","ng","mga","sa","na","ay","ko","mo","niya","kami","kayo","sila",
            "ano","paano","bakit","sino","kailan","saan","kung","pero","kasi",
            "hindi","oo","ho","po","yung","yun","dito","diyan","doon","naman",
            "lang","talaga","sige","eto","yan","ganito","ganyan","galing","mabuti",
            "masaya","salamat","kamusta","musta","anong","yung","rin","ding",
        ])

    def detect(self, text):
        words = set(tokenizer.tokenize(text))
        fil_count = len(words & self.fil_markers)
        return "filipino" if fil_count >= 2 else "english"

class UserProfileTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.messages = []
        self.sentiments = []
        self.interest_counter = Counter()
        self.intent_counter = Counter()
        self.session_start = datetime.now(PH_TZ)

    def update(self, msg, sentiment, topics, intent):
        self.messages.append(msg)
        self.sentiments.append(sentiment["score"])
        for t in topics:
            self.interest_counter[t] += 1
        self.intent_counter[intent] += 1

    def get_insights(self):
        total = len(self.messages)
        if total == 0:
            return {
                "total_interactions": 0,
                "sentiment_trend": "neutral",
                "average_sentiment": 0.5,
                "top_interests": [],
                "dominant_intent": "none",
            }
        avg = sum(self.sentiments) / len(self.sentiments)
        recent = self.sentiments[-5:]
        recent_avg = sum(recent) / len(recent)
        trend = "positive" if recent_avg > 0.60 else "negative" if recent_avg < 0.40 else "neutral"
        top = [t for t, _ in self.interest_counter.most_common(5)]
        dom_intent = self.intent_counter.most_common(1)[0][0] if self.intent_counter else "none"
        return {
            "total_interactions": total,
            "sentiment_trend": trend,
            "average_sentiment": round(avg, 3),
            "top_interests": top,
            "dominant_intent": dom_intent,
        }

KB = {
    "who are you": "I am an AI chatbot built with pure Python Machine Learning: TF-IDF Vectorizer, Naive Bayes, Logistic Regression, KNN, Cosine Similarity, Sentiment Analysis, NER, and Topic Modeling. I work even without internet!",
    "what can you do": "I can answer questions on science, history, coding, math, AI, health, business, and more. I also solve math expressions, detect sentiment, extract topics, and recognize entities — all using pure Python ML!",
    "what is python": "Python is a high-level, general-purpose programming language known for clean and readable syntax. It is used in web development, data science, AI, automation, and more. Created by Guido van Rossum in 1991.",
    "what is machine learning": "Machine Learning (ML) is a branch of AI where computers learn patterns from data to make predictions or decisions — without being explicitly programmed for each task. It includes supervised, unsupervised, and reinforcement learning.",
    "what is deep learning": "Deep Learning uses multi-layer neural networks to learn complex patterns from large datasets. It powers image recognition, speech processing, natural language understanding, and generative AI like GPT.",
    "what is artificial intelligence": "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It covers learning, reasoning, problem-solving, perception, and language understanding. AI includes ML, NLP, computer vision, and robotics.",
    "what is nlp": "Natural Language Processing (NLP) is the field of AI that enables computers to understand, interpret, and generate human language. It powers chatbots, translation, sentiment analysis, and search engines.",
    "what is neural network": "A neural network is a computing system inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process data, learn patterns through backpropagation, and make predictions.",
    "what is flask": "Flask is a lightweight Python web framework used to build web applications and REST APIs quickly. It is minimalist, flexible, and easy to learn — perfect for small to medium projects.",
    "what is django": "Django is a high-level Python web framework that follows the Model-View-Template pattern. It includes built-in authentication, admin panel, ORM, and many features for building robust web applications.",
    "what is html": "HTML (HyperText Markup Language) is the standard markup language for creating web pages. It defines the structure and content using elements like headings, paragraphs, links, images, and forms.",
    "what is css": "CSS (Cascading Style Sheets) controls the visual presentation of HTML pages — including layout, colors, fonts, animations, and responsive design.",
    "what is javascript": "JavaScript is a programming language that makes web pages interactive. It runs in the browser, handles events, manipulates the DOM, and powers modern frontend frameworks like React, Vue, and Angular.",
    "what is api": "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate. REST APIs use HTTP methods (GET, POST, PUT, DELETE) to exchange data, usually in JSON format.",
    "what is rest api": "A REST API is an architectural style for designing networked applications. It uses stateless HTTP requests and standard methods — GET (read), POST (create), PUT (update), DELETE (remove) — to manage resources.",
    "what is database": "A database is an organized collection of structured data stored electronically. SQL databases (MySQL, PostgreSQL) use tables and relations. NoSQL databases (MongoDB, Redis) use documents, key-value pairs, or graphs.",
    "what is algorithm": "An algorithm is a step-by-step set of instructions to solve a problem or perform a task. Examples include sorting (quicksort, mergesort), searching (binary search), and graph traversal (BFS, DFS).",
    "what is data structure": "Data structures organize and store data efficiently. Common types: Arrays, Linked Lists, Stacks, Queues, Trees, Graphs, Hash Tables, and Heaps. Choosing the right one affects program performance.",
    "what is cloud computing": "Cloud computing delivers computing services — servers, storage, databases, networking, software — over the internet. Major providers: AWS, Google Cloud, Azure. Models: IaaS, PaaS, SaaS.",
    "what is cybersecurity": "Cybersecurity protects systems, networks, and data from digital attacks, unauthorized access, and breaches. Includes encryption, firewalls, penetration testing, and incident response.",
    "what is blockchain": "Blockchain is a distributed ledger where data is stored in linked, cryptographically secured blocks across many nodes. It enables trustless, tamper-resistant transactions — used in crypto, NFTs, and supply chains.",
    "what is git": "Git is a distributed version control system that tracks changes in source code. Key commands: git init, git clone, git add, git commit, git push, git pull, git branch, git merge.",
    "what is linux": "Linux is an open-source Unix-like operating system kernel. It powers servers, Android, embedded systems, and supercomputers. Popular distros: Ubuntu, Debian, Fedora, Arch, CentOS.",
    "what is object oriented programming": "Object-Oriented Programming (OOP) organizes code into objects that have attributes (data) and methods (functions). Key principles: Encapsulation, Inheritance, Polymorphism, and Abstraction.",
    "what is recursion": "Recursion is when a function calls itself to solve a smaller version of the same problem. It needs a base case to stop. Used in tree traversal, factorial calculation, and divide-and-conquer algorithms.",
    "what is gravity": "Gravity is the fundamental force that attracts objects with mass toward each other. Newton's law: F = Gm₁m₂/r². Einstein's general relativity describes gravity as the curvature of spacetime caused by mass.",
    "what is photosynthesis": "Photosynthesis is the process plants use to convert sunlight, water (H₂O), and carbon dioxide (CO₂) into glucose (C₆H₁₂O₆) and oxygen (O₂). Equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂.",
    "what is evolution": "Evolution is the process of change in living organisms over generations through natural selection, mutation, genetic drift, and gene flow. Proposed by Charles Darwin in 'On the Origin of Species' (1859).",
    "what is atom": "An atom is the smallest unit of matter that retains element properties. It has a nucleus containing protons (positive) and neutrons (neutral), surrounded by electrons (negative) in orbitals.",
    "what is dna": "DNA (Deoxyribonucleic Acid) is a double-helix molecule that carries genetic instructions for all living organisms. It is made of nucleotides containing a base (A, T, G, C), sugar, and phosphate.",
    "what is black hole": "A black hole is a region in spacetime where gravity is so extreme that nothing — not even light — can escape. Formed when massive stars collapse. Described by the Schwarzschild radius.",
    "what is quantum physics": "Quantum physics studies matter and energy at the atomic/subatomic level. Key concepts: wave-particle duality, superposition, entanglement, and Heisenberg uncertainty principle.",
    "what is relativity": "Einstein's theory of relativity includes Special Relativity (space and time are relative; E=mc²) and General Relativity (gravity is the curvature of spacetime caused by mass and energy).",
    "what is climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns. Primarily caused by burning fossil fuels, deforestation, and industrial emissions that increase greenhouse gases.",
    "what is photon": "A photon is a quantum of electromagnetic radiation — the fundamental particle of light. It has no mass, travels at c = 3×10⁸ m/s, and carries energy E = hf (h = Planck's constant, f = frequency).",
    "what is nanotechnology": "Nanotechnology is the manipulation of matter at the atomic and molecular scale (1-100 nanometers). Applications include medicine (drug delivery), electronics (transistors), and materials science.",
    "what is democracy": "Democracy is a system of government where power is vested in the people, exercised through free elections. Types: direct democracy (citizens vote directly) and representative democracy (elected officials vote).",
    "what is economy": "An economy is the system of production, distribution, and consumption of goods and services. Key concepts: GDP, inflation, unemployment, supply and demand, monetary policy, and fiscal policy.",
    "what is inflation": "Inflation is the rate at which the general price level of goods and services rises, reducing purchasing power. Measured by CPI (Consumer Price Index). Caused by demand-pull, cost-push, or monetary factors.",
    "what is vaccine": "A vaccine trains the immune system to recognize and fight specific pathogens without causing disease. It introduces antigens (weakened/inactivated pathogen or mRNA instructions) to trigger antibody production.",
    "what is virus": "A virus is a microscopic infectious agent that can only replicate inside host cells. It consists of genetic material (DNA or RNA) wrapped in a protein coat (capsid). Causes flu, COVID-19, HIV, and more.",
    "what is photosynthesis": "Photosynthesis converts light energy into chemical energy stored in glucose. Occurs in chloroplasts using chlorophyll. Light reactions produce ATP and NADPH; Calvin cycle fixes CO₂ into sugar.",
    "how are you": "I am running perfectly! All 7 pure Python ML models are loaded: TF-IDF, Naive Bayes, Logistic Regression, KNN, Cosine Similarity, Sentiment Analyzer, and NER. What can I help you with?",
    "what time is it": f"The current time is {datetime.now(PH_TZ).strftime('%I:%M %p, %B %d %Y')} (Philippine Standard Time).",
    "tell me a joke": random.choice([
        "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
        "Why did the developer go broke? Because he used up all his cache! 💸",
        "How many programmers does it take to change a light bulb? None — that is a hardware problem!",
        "I asked my AI model a joke. It returned a 404: Humor Not Found. 😂",
        "A SQL query walks into a bar, walks up to two tables and asks... Can I JOIN you?",
    ]),
    "fun fact": random.choice([
        "The first computer bug was a real moth found in a Harvard Mark II computer relay in 1947!",
        "Python was named after Monty Python's Flying Circus, not the snake!",
        "The first 1GB hard drive (1980) weighed 550 pounds and cost $40,000!",
        "There are more possible chess game variations than atoms in the observable universe.",
        "The word 'robot' comes from the Czech 'robota' meaning forced labor, coined in 1920.",
        "NASA's Voyager 1 code was written in FORTRAN and is still running 45+ years later!",
    ]),
    "what is tfidf": "TF-IDF (Term Frequency-Inverse Document Frequency) is an NLP technique that measures word importance in a document relative to a corpus. TF = word frequency in doc; IDF = log(N/df). Used in search engines and text classification.",
    "what is cosine similarity": "Cosine similarity measures the angle between two vectors in a multi-dimensional space. Formula: cos(θ) = (A·B)/(|A||B|). Value ranges from 0 (completely different) to 1 (identical). Used in document retrieval and recommendation systems.",
    "what is naive bayes": "Naive Bayes is a probabilistic ML classifier based on Bayes' theorem with the 'naive' assumption that features are independent. Despite simplicity, it performs well in text classification, spam detection, and sentiment analysis.",
    "what is logistic regression": "Logistic Regression is an ML algorithm for binary and multiclass classification. It uses the sigmoid/softmax function to output probabilities, trained via gradient descent or SGD to minimize cross-entropy loss.",
    "what is knn": "K-Nearest Neighbors (KNN) is a non-parametric ML algorithm that classifies data points based on the majority label of the K closest training examples, using distance metrics like Euclidean distance.",
    "what is gradient descent": "Gradient Descent is an optimization algorithm that minimizes a loss function by iteratively updating model parameters in the direction of the negative gradient. Variants: Batch GD, Stochastic GD (SGD), Mini-batch GD.",
    "what is overfitting": "Overfitting occurs when an ML model learns the training data too well, including noise, causing poor generalization to new data. Solutions: regularization (L1/L2), dropout, cross-validation, and more training data.",
    "what is regularization": "Regularization prevents overfitting by adding a penalty term to the loss function. L1 (Lasso) adds |weights|, promoting sparsity. L2 (Ridge) adds weights², shrinking coefficients toward zero.",
}

INTENT_CORPUS = {
    "greeting":  [
        "hi","hello","hey","yo","whats up","good morning","good afternoon",
        "good evening","kamusta","musta","sup","howdy","greetings","helo",
        "hi there","hello there","good day","hey there","hiya","wassup",
    ],
    "question":  [
        "what is","how to","why is","when did","where is","who is","explain",
        "define","tell me about","describe","what are","how does","paano",
        "bakit","sino","kailan","saan","anong","ano ang","give me info",
        "can you explain","what does it mean","i want to know","describe to me",
        "what happened","give me an example","what causes","how is made",
    ],
    "math":      [
        "calculate","compute","solve","2 plus 2","multiply","divide",
        "square root","percent of","how much is","add these","subtract",
        "equation","formula","arithmetic","average","algebra","what is 5 times",
        "log of","to the power","what is 10 divided","sin","cos","tan","area",
        "factorial","circumference","what is the value","evaluate this",
    ],
    "coding":    [
        "write a code","write a program","fix my code","debug this",
        "how to code","python script","flask route","api call","sql query",
        "javascript function","html template","css style","code example",
        "create a function","how to install","make a loop","array list",
        "class method","import library","error in code","syntax error",
    ],
    "support":   [
        "help me","i have a problem","not working","error occurred",
        "bug in my","how do i fix","something wrong","issue with my",
        "broken","crashed","failed to run","hindi gumagana","may problema",
        "tulungan mo ako","ayusin mo","what went wrong","i need help",
        "please assist","troubleshoot","cannot run","not responding",
    ],
    "general":   [
        "tell me something","random fact","fun fact","interesting topic",
        "i am bored","entertain me","something cool","surprise me",
        "what do you think","chat with me","joke","trivia","motivate me",
        "give me advice","what should i do","recommend","any suggestion",
        "tell me more","can you help","i wonder","curious about",
    ],
    "closing":   [
        "bye","goodbye","see you","exit","quit","take care","paalam",
        "sige na","salamat","ok thanks bye","until next time","done",
        "that is all","signing off","catch you later","good night",
        "got to go","leaving now","end chat","stop","finish",
    ],
}

class AIEngine:
    def __init__(self):
        print("[AI Engine] Initializing ML models...")

        self.context = deque(maxlen=10)
        self.sent_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.ner = NamedEntityRecognizer()
        self.lang_detect = LanguageDetector()
        self.tracker = UserProfileTracker()
        self.math_solver = MathSolver()

        # --- TF-IDF for intent classification ---
        self.intent_tfidf = TfidfVectorizer(ngram_range=(2,4), analyzer="char_wb", max_features=5000)

        # --- TF-IDF for KB retrieval ---
        self.kb_tfidf = TfidfVectorizer(ngram_range=(1,2), analyzer="word", max_features=4000)

        # --- Build intent training data ---
        X_raw, y_raw = [], []
        for label, samples in INTENT_CORPUS.items():
            for s in samples:
                X_raw.append(tokenizer.normalize(s))
                y_raw.append(label)

        print("[AI Engine] Training TF-IDF + Logistic Regression (intent)...")
        X_intent = self.intent_tfidf.fit_transform(X_raw)

        self.lr_model = LogisticRegression(lr=0.04, max_iter=1000)
        self.lr_model.fit(X_intent, y_raw)

        self.nb_model = NaiveBayes(alpha=0.3)
        self.nb_model.fit(X_intent, y_raw)

        self.knn_model = KNNClassifier(k=3)
        self.knn_model.fit(X_intent, y_raw)

        print("[AI Engine] Building Knowledge Base vectors...")
        self.kb_keys = list(KB.keys())
        self.kb_matrix = self.kb_tfidf.fit_transform([tokenizer.normalize(k) for k in self.kb_keys])

        self.greetings = [
            "Hi there! I am your AI assistant powered by pure Python ML. Ask me anything!",
            "Hello! I am ready to help you. What would you like to know today?",
            "Hey! Great to see you. I can answer questions, solve math, analyze text, and more!",
            "Hi! All my ML models are loaded and ready — Naive Bayes, Logistic Regression, KNN, TF-IDF. Ask away!",
        ]
        self.farewells = [
            "Goodbye! Come back anytime if you have more questions. Take care!",
            "See you later! It was great chatting with you today!",
            "Take care! My ML models will be here waiting whenever you need help.",
        ]
        self.suggestions = {
            "coding":   ["How to create a Flask app?","Write a Python function","What is REST API?"],
            "question": ["Tell me about AI","What is Machine Learning?","What is deep learning?"],
            "math":     ["Calculate 15% of 200","Square root of 144","What is 2^10?"],
            "support":  ["How to fix a Python error?","Why is my code not working?"],
            "greeting": ["What can you do?","Tell me a fun fact","What is TF-IDF?"],
            "general":  ["Tell me a joke","Give me a fun fact","What is cosine similarity?"],
            "closing":  ["Start over","Ask me about science","Ask me about history"],
        }
        print("[AI Engine] All models ready!")

    def _predict_intent(self, text):
        norm = tokenizer.normalize(text)
        vec = self.intent_tfidf.transform([norm])

        lr_probs  = self.lr_model.predict_proba(vec)[0]
        nb_probs  = self.nb_model.predict_proba(vec)[0]
        lr_label  = max(lr_probs, key=lr_probs.get)
        nb_label  = max(nb_probs, key=nb_probs.get)
        knn_label = self.knn_model.predict(vec)[0]

        # Ensemble: majority vote + confidence averaging
        votes = Counter([lr_label, nb_label, knn_label])
        best_label = votes.most_common(1)[0][0]

        lr_conf = lr_probs.get(best_label, 0)
        nb_conf = nb_probs.get(best_label, 0)
        avg_conf = round((lr_conf + nb_conf) / 2, 3)

        return best_label, avg_conf

    def _search_kb(self, text):
        norm = tokenizer.normalize(text)
        vec = self.kb_tfidf.transform([norm])
        idx, score = top_cosine(vec[0], self.kb_matrix)
        if score > 0.20:
            return KB[self.kb_keys[idx]]
        return None

    def _call_gemini(self, msg):
        ctx = "\n".join(list(self.context)[-4:])
        url = (
            "https://generativelanguage.googleapis.com/v1/models/"
            f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": [{"parts": [{"text": (
                "You are a knowledgeable AI assistant. Answer clearly and completely.\n"
                "For math: solve step by step. For code: provide working examples.\n"
                f"Context:\n{ctx}\n\nUser: {msg}"
            )}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
        }
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        return None

    def _offline_fallback(self, intent, entities):
        entity_str = ""
        for etype, terms in entities.items():
            if terms:
                entity_str += f" about {', '.join(terms)}"
        fallbacks = {
            "math":    f"I detected a math question{entity_str}! Try formats like: '15% of 200', '25 * 4', 'square root of 81', 'sin(30)', 'log(100)'.",
            "coding":  f"That seems like a coding question{entity_str}. My offline KB has basics — try asking about Python, Flask, HTML, CSS, JavaScript, or API concepts!",
            "support": f"Looks like you need technical help{entity_str}. Describe the error message or problem in detail and I will try my best!",
            "question": f"I do not have that specific answer offline{entity_str}. Try asking about Python, AI, ML, science, history, math, or programming!",
            "general": f"Interesting topic{entity_str}! I am offline right now but you can ask me about technology, science, coding, math, or history!",
        }
        return fallbacks.get(intent, f"I am not sure about that{entity_str}. Try asking about technology, science, math, or coding!")

    def process(self, msg):
        self.context.append(f"User: {msg}")

        # --- Run all NLP pipelines ---
        sentiment   = self.sent_analyzer.analyze(msg)
        topics      = self.topic_modeler.get_topics(msg)
        entities    = self.ner.recognize(msg)
        language    = self.lang_detect.detect(msg)
        intent, conf = self._predict_intent(msg)
        self.tracker.update(msg, sentiment, topics, intent)
        user_insights = self.tracker.get_insights()
        sugg = self.suggestions.get(intent, ["Ask me anything!", "Tell me a fun fact", "What is ML?"])

        # ===== DECISION PIPELINE =====

        # 1. TIME
        if any(w in msg.lower() for w in ["time","oras","anong oras","what time","current time"]):
            now = datetime.now(PH_TZ)
            reply = f"The current time is {now.strftime('%I:%M %p, %B %d %Y')} (Philippine Standard Time)."

        # 2. GREETING
        elif intent == "greeting" and conf > 0.40:
            reply = random.choice(self.greetings)

        # 3. FAREWELL
        elif intent == "closing" and conf > 0.40:
            reply = random.choice(self.farewells)

        # 4. MATH (pure Python solver)
        else:
            math_result = self.math_solver.solve(msg)
            if math_result:
                reply = math_result

            # 5. KNOWLEDGE BASE (TF-IDF Cosine Similarity retrieval)
            else:
                kb_result = self._search_kb(msg)
                if kb_result:
                    reply = kb_result

                # 6. GEMINI API (cloud backup)
                else:
                    gemini_result = self._call_gemini(msg)
                    if gemini_result:
                        reply = gemini_result

                    # 7. OFFLINE ML FALLBACK
                    else:
                        reply = self._offline_fallback(intent, entities)

        self.context.append(f"Assistant: {reply}")

        return {
            "message": reply,
            "ml_insights": {
                "intent": intent,
                "confidence": conf,
                "sentiment": sentiment,
                "topics": topics,
                "entities": entities,
                "language": language,
                "suggestions": sugg,
                "user_insights": user_insights,
            }
        }

engine = AIEngine()

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").strip() if data else ""
    if not msg:
        return jsonify({"status": "error", "message": "Please type a message."})
    result = engine.process(msg)
    return jsonify({"status": "success", **result})
@app.route("/clear", methods=["POST"])
def clear():
    engine.context.clear()
    engine.tracker.reset()
    return jsonify({"status": "success"})
@app.route("/insights", methods=["GET"])
def insights():
    data = engine.tracker.get_insights()
    if data["total_interactions"] == 0:
        return jsonify({"status": "empty"})
    return jsonify({"status": "success", "insights": data})
@app.route("/export", methods=["GET"])
def export_data():
    return jsonify({
        "status": "success",
        "data": {
            "insights": engine.tracker.get_insights(),
            "context_length": len(engine.context),
            "exported_at": datetime.now(PH_TZ).strftime("%Y-%m-%d %H:%M:%S PST"),
        }
    })
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip() if data else ""
    if not text:
        return jsonify({"status": "error", "message": "No text provided."})
    return jsonify({
        "status": "success",
        "sentiment": engine.sent_analyzer.analyze(text),
        "topics": engine.topic_modeler.get_topics(text),
        "entities": engine.ner.recognize(text),
        "language": engine.lang_detect.detect(text),
        "intent": engine._predict_intent(text)[0],
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)