import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Stores performance metrics for each task"""
    time_taken: float
    accuracy: float
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    task_specific_score: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate weighted overall score (time + accuracy)"""
        # Lower time is better, normalize to 0-1 scale
        time_score = 1.0 / (1.0 + self.time_taken)
        # Weighted combination: 70% accuracy, 30% speed
        return 0.7 * self.accuracy + 0.3 * time_score


@dataclass
class CompetitionResult:
    """Stores complete results for an algorithm"""
    algorithm_name: str
    classification_metrics: PerformanceMetrics
    qa_metrics: PerformanceMetrics
    summarization_metrics: PerformanceMetrics
    total_score: float
    explanation: str
    failure_points: List[str]


class BaseLearningAlgorithm(ABC):
    """Abstract base class for all competing algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.training_time = 0.0
        self.failure_points = []
        
    @abstractmethod
    def train_classifier(self, X_train, y_train):
        """Train classification model"""
        pass
    
    @abstractmethod
    def predict_class(self, X_test):
        """Predict classes"""
        pass
    
    @abstractmethod
    def answer_question(self, document: str, question: str) -> str:
        """Answer question from document"""
        pass
    
    @abstractmethod
    def summarize(self, document: str, max_length: int = 100) -> str:
        """Summarize document"""
        pass
    
    @abstractmethod
    def explain_technique(self) -> str:
        """Explain the learning technique used"""
        pass
    
    def record_failure(self, task: str, reason: str):
        """Record failure points"""
        self.failure_points.append(f"{task}: {reason}")


class NaiveBayesLearner(BaseLearningAlgorithm):
    """Naive Bayes based learning algorithm"""
    
    def __init__(self):
        super().__init__("Naive Bayes Learner")
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = MultinomialNB()
        self.vocab = None
        
    def train_classifier(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        self.vocab = self.vectorizer.get_feature_names_out()
        
    def predict_class(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_vec)
    
    def answer_question(self, document: str, question: str) -> str:
        """Simple keyword-based QA"""
        try:
            doc_sentences = document.split('.')
            question_lower = question.lower()
            
            # Find most relevant sentence
            best_sentence = ""
            max_overlap = 0
            
            for sentence in doc_sentences:
                sentence_lower = sentence.lower()
                overlap = sum(1 for word in question_lower.split() 
                            if word in sentence_lower and len(word) > 3)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence.strip()
            
            return best_sentence if best_sentence else "Answer not found"
        except Exception as e:
            self.record_failure("QA", f"Keyword matching failed: {str(e)}")
            return "Error in answering"
    
    def summarize(self, document: str, max_length: int = 100) -> str:
        """Extractive summarization using TF-IDF"""
        try:
            sentences = [s.strip() for s in document.split('.') if s.strip()]
            if len(sentences) <= 2:
                return document
            
            # Vectorize sentences
            vectorizer = TfidfVectorizer()
            sentence_vectors = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF values)
            scores = np.array(sentence_vectors.sum(axis=1)).flatten()
            
            # Get top sentences
            num_sentences = min(3, len(sentences))
            top_indices = scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary = '. '.join([sentences[i] for i in top_indices]) + '.'
            return summary[:max_length]
        except Exception as e:
            self.record_failure("Summarization", f"TF-IDF scoring failed: {str(e)}")
            return document[:max_length]
    
    def explain_technique(self) -> str:
        return """
        Technique: Naive Bayes with TF-IDF Vectorization
        
        Classification: Uses TF-IDF to convert text to numerical features, 
        then applies Multinomial Naive Bayes which assumes feature independence.
        Works well with high-dimensional sparse data.
        
        QA: Keyword-based matching - finds sentences with highest word overlap 
        with the question.
        
        Summarization: Extractive approach using TF-IDF scores to identify 
        most important sentences.
        """


class LogisticRegressionLearner(BaseLearningAlgorithm):
    """Logistic Regression based learning algorithm"""
    
    def __init__(self):
        super().__init__("Logistic Regression Learner")
        self.vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        
    def train_classifier(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        
    def predict_class(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_vec)
    
    def answer_question(self, document: str, question: str) -> str:
        """N-gram based QA"""
        try:
            doc_sentences = document.split('.')
            question_words = set(question.lower().split())
            
            best_sentence = ""
            max_score = 0
            
            for sentence in doc_sentences:
                sentence_words = set(sentence.lower().split())
                # Jaccard similarity
                if len(question_words) > 0:
                    score = len(question_words & sentence_words) / len(question_words | sentence_words)
                    if score > max_score:
                        max_score = score
                        best_sentence = sentence.strip()
            
            return best_sentence if best_sentence else "Answer not found"
        except Exception as e:
            self.record_failure("QA", f"N-gram matching failed: {str(e)}")
            return "Error in answering"
    
    def summarize(self, document: str, max_length: int = 100) -> str:
        """Position-weighted extractive summarization"""
        try:
            sentences = [s.strip() for s in document.split('.') if s.strip()]
            if len(sentences) <= 2:
                return document
            
            # Weight by position (earlier sentences more important)
            scores = []
            for i, sent in enumerate(sentences):
                position_weight = 1.0 / (i + 1)
                length_weight = len(sent.split()) / 20  # Prefer medium length
                scores.append(position_weight + length_weight)
            
            num_sentences = min(3, len(sentences))
            top_indices = np.argsort(scores)[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary = '. '.join([sentences[i] for i in top_indices]) + '.'
            return summary[:max_length]
        except Exception as e:
            self.record_failure("Summarization", f"Position weighting failed: {str(e)}")
            return document[:max_length]
    
    def explain_technique(self) -> str:
        return """
        Technique: Logistic Regression with N-gram Features
        
        Classification: Uses Count Vectorizer with unigrams and bigrams, 
        then applies logistic regression. Captures word sequences better than 
        single words.
        
        QA: Jaccard similarity between question and sentences - measures 
        overlap as proportion of unique words.
        
        Summarization: Position-weighted extraction - prioritizes earlier 
        sentences and medium-length sentences.
        """


class RandomForestLearner(BaseLearningAlgorithm):
    """Random Forest based learning algorithm"""
    
    def __init__(self):
        super().__init__("Random Forest Learner")
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train_classifier(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        
    def predict_class(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_test_vec)
    
    def answer_question(self, document: str, question: str) -> str:
        """Sentence ranking based QA"""
        try:
            doc_sentences = document.split('.')
            question_vec = self.vectorizer.transform([question])
            
            best_sentence = ""
            max_similarity = -1
            
            for sentence in doc_sentences:
                if sentence.strip():
                    sent_vec = self.vectorizer.transform([sentence])
                    # Cosine similarity
                    similarity = (question_vec.multiply(sent_vec).sum()) / (
                        np.sqrt(question_vec.multiply(question_vec).sum()) * 
                        np.sqrt(sent_vec.multiply(sent_vec).sum()) + 1e-10
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_sentence = sentence.strip()
            
            return best_sentence if best_sentence else "Answer not found"
        except Exception as e:
            self.record_failure("QA", f"Vector similarity failed: {str(e)}")
            return "Error in answering"
    
    def summarize(self, document: str, max_length: int = 100) -> str:
        """Diversity-based extractive summarization"""
        try:
            sentences = [s.strip() for s in document.split('.') if s.strip()]
            if len(sentences) <= 2:
                return document
            
            # Vectorize all sentences
            sent_vectors = self.vectorizer.transform(sentences).toarray()
            
            # Select diverse sentences using greedy approach
            selected = [0]  # Start with first sentence
            
            while len(selected) < min(3, len(sentences)):
                max_min_distance = -1
                next_idx = -1
                
                for i in range(len(sentences)):
                    if i not in selected:
                        # Calculate minimum distance to selected sentences
                        min_dist = min([np.linalg.norm(sent_vectors[i] - sent_vectors[j]) 
                                       for j in selected])
                        if min_dist > max_min_distance:
                            max_min_distance = min_dist
                            next_idx = i
                
                if next_idx != -1:
                    selected.append(next_idx)
                else:
                    break
            
            selected = sorted(selected)
            summary = '. '.join([sentences[i] for i in selected]) + '.'
            return summary[:max_length]
        except Exception as e:
            self.record_failure("Summarization", f"Diversity selection failed: {str(e)}")
            return document[:max_length]
    
    def explain_technique(self) -> str:
        return """
        Technique: Random Forest with TF-IDF Features
        
        Classification: Ensemble of decision trees with TF-IDF features. 
        Reduces overfitting through voting and handles non-linear patterns well.
        
        QA: Cosine similarity between TF-IDF vectors - measures semantic 
        similarity between question and sentences.
        
        Summarization: Diversity-based extraction - selects sentences that 
        are maximally different from each other to cover more content.
        """


class DocumentLearningArena:
    """Main competition arena for algorithms"""
    
    def __init__(self):
        self.algorithms: List[BaseLearningAlgorithm] = []
        self.results: List[CompetitionResult] = []
        
    def register_algorithm(self, algorithm: BaseLearningAlgorithm):
        """Register a new algorithm to compete"""
        self.algorithms.append(algorithm)
        print(f"✓ Registered: {algorithm.name}")
        
    def evaluate_classification(self, algorithm: BaseLearningAlgorithm, 
                               X_train, X_test, y_train, y_test) -> PerformanceMetrics:
        """Evaluate classification performance"""
        start_time = time.time()
        
        try:
            algorithm.train_classifier(X_train, y_train)
            predictions = algorithm.predict_class(X_test)
            
            time_taken = time.time() - start_time
            
            metrics = PerformanceMetrics(
                time_taken=time_taken,
                accuracy=accuracy_score(y_test, predictions),
                precision=precision_score(y_test, predictions, average='weighted', zero_division=0),
                recall=recall_score(y_test, predictions, average='weighted', zero_division=0),
                f1_score=f1_score(y_test, predictions, average='weighted', zero_division=0)
            )
            return metrics
        except Exception as e:
            algorithm.record_failure("Classification", str(e))
            return PerformanceMetrics(time_taken=999.0, accuracy=0.0)
    
    def evaluate_qa(self, algorithm: BaseLearningAlgorithm, 
                    documents: List[str], qa_pairs: List[Tuple[str, str]]) -> PerformanceMetrics:
        """Evaluate QA performance"""
        start_time = time.time()
        correct = 0
        total = len(qa_pairs)
        
        try:
            for document, (question, answer) in zip(documents, qa_pairs):
                predicted_answer = algorithm.answer_question(document, question)
                # Simple containment check
                if answer.lower() in predicted_answer.lower():
                    correct += 1
            
            time_taken = time.time() - start_time
            accuracy = correct / total if total > 0 else 0.0
            
            return PerformanceMetrics(
                time_taken=time_taken,
                accuracy=accuracy,
                task_specific_score=accuracy
            )
        except Exception as e:
            algorithm.record_failure("QA", str(e))
            return PerformanceMetrics(time_taken=999.0, accuracy=0.0)
    
    def evaluate_summarization(self, algorithm: BaseLearningAlgorithm,
                              documents: List[str], reference_summaries: List[str]) -> PerformanceMetrics:
        """Evaluate summarization performance"""
        start_time = time.time()
        scores = []
        
        try:
            for document, reference in zip(documents, reference_summaries):
                summary = algorithm.summarize(document, max_length=100)
                
                # Simple ROUGE-like scoring (word overlap)
                ref_words = set(reference.lower().split())
                sum_words = set(summary.lower().split())
                
                if len(ref_words) > 0:
                    precision = len(ref_words & sum_words) / len(sum_words) if len(sum_words) > 0 else 0
                    recall = len(ref_words & sum_words) / len(ref_words)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    scores.append(f1)
                else:
                    scores.append(0.0)
            
            time_taken = time.time() - start_time
            avg_score = np.mean(scores) if scores else 0.0
            
            return PerformanceMetrics(
                time_taken=time_taken,
                accuracy=avg_score,
                task_specific_score=avg_score
            )
        except Exception as e:
            algorithm.record_failure("Summarization", str(e))
            return PerformanceMetrics(time_taken=999.0, accuracy=0.0)
    
    def run_competition(self, classification_data: Dict, qa_data: Dict, 
                       summarization_data: Dict) -> CompetitionResult:
        """Run full competition across all tasks"""
        print("\n" + "="*60)
        print("🏁 STARTING DOCUMENT LEARNING COMPETITION 🏁")
        print("="*60 + "\n")
        
        self.results = []
        
        for algorithm in self.algorithms:
            print(f"\n📊 Testing: {algorithm.name}")
            print("-" * 40)
            
            # Classification
            print("  ⚡ Classification...", end=" ")
            clf_metrics = self.evaluate_classification(
                algorithm,
                classification_data['X_train'],
                classification_data['X_test'],
                classification_data['y_train'],
                classification_data['y_test']
            )
            print(f"✓ (Acc: {clf_metrics.accuracy:.3f}, Time: {clf_metrics.time_taken:.3f}s)")
            
            # QA
            print("  ⚡ Question Answering...", end=" ")
            qa_metrics = self.evaluate_qa(
                algorithm,
                qa_data['documents'],
                qa_data['qa_pairs']
            )
            print(f"✓ (Acc: {qa_metrics.accuracy:.3f}, Time: {qa_metrics.time_taken:.3f}s)")
            
            # Summarization
            print("  ⚡ Summarization...", end=" ")
            sum_metrics = self.evaluate_summarization(
                algorithm,
                summarization_data['documents'],
                summarization_data['references']
            )
            print(f"✓ (Score: {sum_metrics.accuracy:.3f}, Time: {sum_metrics.time_taken:.3f}s)")
            
            # Calculate total score
            total_score = (clf_metrics.overall_score() + 
                          qa_metrics.overall_score() + 
                          sum_metrics.overall_score()) / 3
            
            result = CompetitionResult(
                algorithm_name=algorithm.name,
                classification_metrics=clf_metrics,
                qa_metrics=qa_metrics,
                summarization_metrics=sum_metrics,
                total_score=total_score,
                explanation=algorithm.explain_technique(),
                failure_points=algorithm.failure_points
            )
            
            self.results.append(result)
        
        return self.get_winner()
    
    def get_winner(self) -> CompetitionResult:
        """Get the winning algorithm"""
        if not self.results:
            raise ValueError("No results available")
        
        winner = max(self.results, key=lambda x: x.total_score)
        return winner
    
    def display_results(self):
        """Display competition results"""
        print("\n" + "="*60)
        print("🏆 COMPETITION RESULTS 🏆")
        print("="*60 + "\n")
        
        # Sort by total score
        sorted_results = sorted(self.results, key=lambda x: x.total_score, reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            print(f"{medal} {result.algorithm_name}")
            print(f"   Total Score: {result.total_score:.4f}")
            print(f"   Classification: Acc={result.classification_metrics.accuracy:.3f}, "
                  f"Time={result.classification_metrics.time_taken:.3f}s")
            print(f"   QA: Acc={result.qa_metrics.accuracy:.3f}, "
                  f"Time={result.qa_metrics.time_taken:.3f}s")
            print(f"   Summarization: Score={result.summarization_metrics.accuracy:.3f}, "
                  f"Time={result.summarization_metrics.time_taken:.3f}s")
            print()
        
        # Winner explanation
        winner = sorted_results[0]
        print("\n" + "="*60)
        print(f"🎯 WINNER: {winner.algorithm_name}")
        print("="*60)
        print("\n📖 TECHNIQUE EXPLANATION:")
        print(winner.explanation)
        
        if winner.failure_points:
            print("\n⚠️  FAILURE POINTS:")
            for failure in winner.failure_points:
                print(f"  - {failure}")
        else:
            print("\n✨ No failures recorded!")
        
        print("\n" + "="*60)


# Example usage
if __name__ == "__main__":
    # Sample data for demonstration
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by biological neural networks.",
        "Deep learning uses multiple layers to learn representations.",
        "Natural language processing deals with text and speech.",
    ]
    sample_labels = ['AI', 'Programming', 'AI', 'AI', 'NLP']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sample_texts, sample_labels, test_size=0.4, random_state=42
    )
    
    # QA data
    qa_documents = [
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Python is widely used in data science due to its simplicity and powerful libraries."
    ]
    qa_pairs = [
        ("What is machine learning?", "subset of artificial intelligence"),
        ("Why is Python popular?", "simplicity and powerful libraries")
    ]
    
    # Summarization data
    sum_documents = [
        "Machine learning is revolutionizing technology. It enables computers to learn from data. "
        "Applications range from image recognition to natural language processing. "
        "The field continues to grow rapidly with new innovations.",
    ]
    sum_references = [
        "Machine learning enables computers to learn from data and is revolutionizing technology."
    ]
    
    # Prepare data
    classification_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    qa_data = {
        'documents': qa_documents,
        'qa_pairs': qa_pairs
    }
    
    summarization_data = {
        'documents': sum_documents,
        'references': sum_references
    }
    
    # Create arena and register algorithms
    arena = DocumentLearningArena()
    arena.register_algorithm(NaiveBayesLearner())
    arena.register_algorithm(LogisticRegressionLearner())
    arena.register_algorithm(RandomForestLearner())
    
    # Run competition
    winner = arena.run_competition(classification_data, qa_data, summarization_data)
    
    # Display results
    arena.display_results()