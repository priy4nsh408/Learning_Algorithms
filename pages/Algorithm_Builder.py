import streamlit as st
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Algorithm Builder", page_icon="💻", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .code-editor {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    .output-box {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        min-height: 200px;
        max-height: 400px;
        overflow-y: auto;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("💻 Algorithm Builder")
st.markdown("Build and test your own custom learning algorithms with our interactive Python compiler.")

# Initialize session state
if 'code' not in st.session_state:
    st.session_state.code = ""
if 'output' not in st.session_state:
    st.session_state.output = ""
if 'test_results' not in st.session_state:
    st.session_state.test_results = None

# Sidebar
st.sidebar.header("🎓 Learning Resources")

with st.sidebar.expander("📖 Getting Started"):
    st.markdown("""
    **Quick Start:**
    1. Choose a template or start from scratch
    2. Write your algorithm code
    3. Test it on sample data
    4. Submit to competition!
    
    **Requirements:**
    - Inherit from `BaseLearningAlgorithm`
    - Implement all required methods
    - Handle errors gracefully
    """)

with st.sidebar.expander("🔧 Available Libraries"):
    st.code("""
# ML Libraries
import numpy as np
import sklearn
import pandas as pd

# NLP Libraries  
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom
from competition_module import BaseLearningAlgorithm
    """, language='python')

with st.sidebar.expander("📝 Code Templates"):
    template_choice = st.selectbox(
        "Choose a template:",
        ["Blank", "Basic Classifier", "Advanced QA", "Complete Algorithm"]
    )

# Template definitions
templates = {
    "Blank": "",
    "Basic Classifier": '''from competition_module import BaseLearningAlgorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class MyCustomLearner(BaseLearningAlgorithm):
    """Your custom learning algorithm"""
    
    def __init__(self):
        super().__init__("My Custom Algorithm")
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        
    def train_classifier(self, X_train, y_train):
        X_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_vec, y_train)
        
    def predict_class(self, X_test):
        X_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_vec)
        
    def answer_question(self, document: str, question: str) -> str:
        # TODO: Implement your QA logic
        return "Answer not implemented"
        
    def summarize(self, document: str, max_length: int = 100) -> str:
        # TODO: Implement your summarization logic
        return document[:max_length]
        
    def explain_technique(self) -> str:
        return "Your algorithm explanation here"

# Test your algorithm
learner = MyCustomLearner()
print(f"Created: {learner.name}")
''',
    "Advanced QA": '''from competition_module import BaseLearningAlgorithm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class AdvancedQALearner(BaseLearningAlgorithm):
    """Advanced QA with semantic similarity"""
    
    def __init__(self):
        super().__init__("Advanced QA Algorithm")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        
    def train_classifier(self, X_train, y_train):
        # Simple implementation
        pass
        
    def predict_class(self, X_test):
        # Simple implementation
        return ["Unknown"] * len(X_test)
        
    def answer_question(self, document: str, question: str) -> str:
        """Advanced semantic similarity QA"""
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        
        if not sentences:
            return "No content found"
        
        # Vectorize
        all_text = [question] + sentences
        vectors = self.vectorizer.fit_transform(all_text)
        
        # Calculate cosine similarity
        question_vec = vectors[0].toarray()
        sentence_vecs = vectors[1:].toarray()
        
        similarities = np.dot(sentence_vecs, question_vec.T).flatten()
        best_idx = np.argmax(similarities)
        
        return sentences[best_idx]
        
    def summarize(self, document: str, max_length: int = 100) -> str:
        return document[:max_length]
        
    def explain_technique(self) -> str:
        return "Uses TF-IDF and cosine similarity for QA"

# Test
learner = AdvancedQALearner()
test_doc = "Python is great. It has many libraries. Machine learning is powerful."
answer = learner.answer_question(test_doc, "What is Python?")
print(f"Answer: {answer}")
''',
    "Complete Algorithm": '''from competition_module import BaseLearningAlgorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class CompleteCustomLearner(BaseLearningAlgorithm):
    """Complete implementation with all methods"""
    
    def __init__(self):
        super().__init__("Complete Custom Algorithm")
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.classifier = GradientBoostingClassifier(n_estimators=50)
        self.is_trained = False
        
    def train_classifier(self, X_train, y_train):
        try:
            X_vec = self.vectorizer.fit_transform(X_train)
            self.classifier.fit(X_vec, y_train)
            self.is_trained = True
            print(f"✓ Training complete: {len(X_train)} samples")
        except Exception as e:
            self.record_failure("Training", str(e))
            raise
        
    def predict_class(self, X_test):
        if not self.is_trained:
            raise ValueError("Model not trained!")
        X_vec = self.vectorizer.transform(X_test)
        return self.classifier.predict(X_vec)
        
    def answer_question(self, document: str, question: str) -> str:
        """Hybrid keyword + semantic QA"""
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        
        if not sentences:
            return "No content"
        
        question_words = set(question.lower().split())
        best_score = 0
        best_sentence = sentences[0]
        
        for sentence in sentences:
            # Keyword overlap
            sent_words = set(sentence.lower().split())
            overlap = len(question_words & sent_words)
            
            # Length bonus
            length_score = min(len(sentence.split()) / 15, 1.0)
            
            score = overlap + length_score
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        return best_sentence
        
    def summarize(self, document: str, max_length: int = 100) -> str:
        """Smart extractive summarization"""
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return document[:max_length]
        
        # Score sentences
        scores = []
        for i, sent in enumerate(sentences):
            # Position weight
            pos_weight = 1.0 / (i + 1)
            # Length weight
            len_weight = min(len(sent.split()) / 20, 1.0)
            # Keyword density (rough measure)
            keyword_weight = len([w for w in sent.split() if len(w) > 6]) / max(len(sent.split()), 1)
            
            score = pos_weight * 0.5 + len_weight * 0.3 + keyword_weight * 0.2
            scores.append(score)
        
        # Get top sentences
        num_sents = min(3, len(sentences))
        top_indices = sorted(np.argsort(scores)[-num_sents:])
        
        summary = '. '.join([sentences[i] for i in top_indices]) + '.'
        return summary[:max_length]
        
    def explain_technique(self) -> str:
        return """
        Complete Custom Algorithm using Gradient Boosting
        
        Classification: Gradient Boosting with TF-IDF and N-grams
        QA: Hybrid keyword overlap + length scoring
        Summarization: Multi-factor sentence scoring (position + length + keywords)
        
        Strengths: Balanced accuracy and speed, handles diverse text well
        """

# Test the algorithm
if __name__ == "__main__":
    learner = CompleteCustomLearner()
    print(f"Algorithm: {learner.name}")
    
    # Test classification
    X_train = ["AI is amazing", "Python is great", "ML is powerful"]
    y_train = ["Tech", "Programming", "Tech"]
    
    learner.train_classifier(X_train, y_train)
    predictions = learner.predict_class(["Artificial intelligence rocks"])
    print(f"Prediction: {predictions[0]}")
    
    # Test QA
    doc = "Python is versatile. It is used for web development and data science."
    answer = learner.answer_question(doc, "What is Python used for?")
    print(f"QA Answer: {answer}")
    
    print("\\n✅ All tests passed!")
'''
}

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Code Editor")
    
    # Template loader
    if st.button("📋 Load Template", use_container_width=True):
        st.session_state.code = templates[template_choice]
        st.rerun()
    
    # Code editor
    code = st.text_area(
        "Write your algorithm here:",
        value=st.session_state.code,
        height=500,
        key="code_editor",
        help="Write your custom learning algorithm. Inherit from BaseLearningAlgorithm and implement required methods."
    )
    
    st.session_state.code = code
    
    # Action buttons
    col_run, col_clear, col_save = st.columns(3)
    
    with col_run:
        run_button = st.button("▶️ Run Code", type="primary", use_container_width=True)
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.code = ""
            st.session_state.output = ""
            st.rerun()
    
    with col_save:
        if st.button("💾 Save", use_container_width=True):
            # Save to file
            with open("my_algorithm.py", "w") as f:
                f.write(st.session_state.code)
            st.success("✅ Saved to my_algorithm.py")

with col2:
    st.subheader("📤 Output Console")
    
    if run_button:
        if not code.strip():
            st.warning("⚠️ Please write some code first!")
        else:
            # Capture output
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            try:
                # Redirect stdout and stderr
                with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                    # Execute code
                    exec_globals = {
                        '__name__': '__main__',
                        'np': __import__('numpy'),
                        'pd': __import__('pandas'),
                        'sklearn': __import__('sklearn'),
                    }
                    
                    # Try to import competition module
                    try:
                        exec("from competition_module import BaseLearningAlgorithm", exec_globals)
                    except:
                        st.warning("⚠️ competition_module not found. Some features may not work.")
                    
                    exec(code, exec_globals)
                
                # Get output
                stdout_output = output_buffer.getvalue()
                stderr_output = error_buffer.getvalue()
                
                if stderr_output:
                    st.error("❌ Errors:")
                    st.code(stderr_output, language='text')
                
                if stdout_output:
                    st.success("✅ Output:")
                    st.code(stdout_output, language='text')
                else:
                    st.info("ℹ️ Code executed successfully with no output.")
                
                st.session_state.output = stdout_output + stderr_output
                
            except Exception as e:
                error_msg = f"❌ Error:\n{traceback.format_exc()}"
                st.error(error_msg)
                st.session_state.output = error_msg
    
    # Display saved output
    elif st.session_state.output:
        st.code(st.session_state.output, language='text')
    else:
        st.info("👈 Click 'Run Code' to see output here")

# Testing section
st.markdown("---")
st.subheader("🧪 Test Your Algorithm")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Test Data")
    
    test_type = st.selectbox(
        "Select test type:",
        ["Classification", "Question Answering", "Summarization"]
    )
    
    if test_type == "Classification":
        st.code("""
# Sample classification data
X_train = [
    "Machine learning is powerful",
    "Python is a great language",
    "AI is the future"
]
y_train = ["Tech", "Programming", "Tech"]

X_test = ["Artificial intelligence is amazing"]
        """, language='python')
    
    elif test_type == "Question Answering":
        st.code("""
# Sample QA data
document = '''
Machine learning is a branch of AI. 
It enables computers to learn from data.
Applications include image recognition and NLP.
'''

question = "What is machine learning?"
expected_answer = "branch of AI"
        """, language='python')
    
    else:
        st.code("""
# Sample summarization data
document = '''
Machine learning is revolutionizing technology. 
It enables computers to learn from data without 
explicit programming. Applications range from 
image recognition to natural language processing.
The field continues to grow rapidly.
'''

max_length = 100
        """, language='python')

with col2:
    st.markdown("#### Quick Test")
    
    if st.button("🧪 Run Quick Test", use_container_width=True):
        st.info("Feature coming soon! Integrate your algorithm with the competition arena.")

# Documentation
st.markdown("---")
st.subheader("📚 Documentation")

tab1, tab2, tab3 = st.tabs(["📖 API Reference", "💡 Examples", "❓ FAQ"])

with tab1:
    st.markdown("""
    ### BaseLearningAlgorithm API
    
    Your custom algorithm must inherit from `BaseLearningAlgorithm` and implement these methods:
    
    ```python
    class BaseLearningAlgorithm(ABC):
        def __init__(self, name: str):
            '''Initialize with algorithm name'''
            
        @abstractmethod
        def train_classifier(self, X_train, y_train):
            '''Train classification model'''
            
        @abstractmethod
        def predict_class(self, X_test):
            '''Predict classes for test data'''
            
        @abstractmethod
        def answer_question(self, document: str, question: str) -> str:
            '''Answer question based on document'''
            
        @abstractmethod
        def summarize(self, document: str, max_length: int = 100) -> str:
            '''Generate document summary'''
            
        @abstractmethod
        def explain_technique(self) -> str:
            '''Explain your algorithm's approach'''
            
        def record_failure(self, task: str, reason: str):
            '''Record failure points for analysis'''
    ```
    """)

with tab2:
    st.markdown("""
    ### Example: Custom Vectorizer
    
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    
    class MyLearner(BaseLearningAlgorithm):
        def __init__(self):
            super().__init__("My Algorithm")
            # Custom vectorizer with character n-grams
            self.vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                max_features=1000
            )
    ```
    
    ### Example: Error Handling
    
    ```python
    def train_classifier(self, X_train, y_train):
        try:
            X_vec = self.vectorizer.fit_transform(X_train)
            self.classifier.fit(X_vec, y_train)
        except Exception as e:
            self.record_failure("Training", str(e))
            raise
    ```
    
    ### Example: Custom Scoring
    
    ```python
    def summarize(self, document: str, max_length: int = 100) -> str:
        sentences = document.split('.')
        
        # Custom scoring function
        def score_sentence(sent, position):
            keywords = ['important', 'key', 'main']
            keyword_bonus = sum(1 for k in keywords if k in sent.lower())
            position_weight = 1.0 / (position + 1)
            return keyword_bonus + position_weight
        
        scores = [score_sentence(s, i) for i, s in enumerate(sentences)]
        # Select top sentences...
    ```
    """)

with tab3:
    st.markdown("""
    ### Frequently Asked Questions
    
    **Q: What libraries can I use?**
    
    A: You can use numpy, pandas, scikit-learn, and nltk. Most standard ML libraries are available.
    
    **Q: How do I test my algorithm?**
    
    A: Click "Run Code" to execute. Then go to the Competition Arena to test against other algorithms.
    
    **Q: Can I save my algorithm?**
    
    A: Yes! Click the "Save" button to save your code as a Python file.
    
    **Q: What if my code has errors?**
    
    A: Error messages will appear in the Output Console. Check the syntax and try again.
    
    **Q: How do I make my algorithm faster?**
    
    A: Reduce feature dimensions, use simpler models, or optimize vector operations.
    
    **Q: Can I use deep learning?**
    
    A: Currently, only scikit-learn models are supported. Deep learning support coming soon!
    """)

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📚 View Algorithms", use_container_width=True):
        st.switch_page("pages/1_📚_Algorithm_Explorer.py")

with col2:
    if st.button("⚔️ Test in Arena", use_container_width=True):
        st.switch_page("pages/2_⚔️_Competition_Arena.py")

with col3:
    st.download_button(
        label="📥 Download Code",
        data=st.session_state.code,
        file_name="my_algorithm.py",
        mime="text/plain",
        use_container_width=True
    )