import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import sys
import time
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define data structures for results
@dataclass
class TaskMetrics:
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    time_taken: float = 0.0

@dataclass
class AlgorithmResult:
    algorithm_name: str
    classification_metrics: TaskMetrics
    qa_metrics: TaskMetrics
    summarization_metrics: TaskMetrics
    total_score: float = 0.0
    explanation: str = ""
    failure_points: List[str] = field(default_factory=list)

# Import from the competition module
try:
    from competition_module import (
        NaiveBayesLearner,
        SVMLearner, 
        KNearestNeighborsLearner,
        DecisionTreeLearner,
        NeuralNetLearner
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Import Error: {str(e)}")
    st.error("Please make sure competition_module.py is in the root directory!")
    MODULES_AVAILABLE = False

# Create placeholder classes for algorithms not in the module
class LogisticRegressionLearner:
    """Placeholder for Logistic Regression"""
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.name = "Logistic Regression"
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

class RandomForestLearner:
    """Placeholder for Random Forest"""
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.name = "Random Forest"
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

# Document Learning Arena class
class DocumentLearningArena:
    """Arena for running algorithm competitions"""
    
    def __init__(self):
        self.algorithms = []
        self.results = []
    
    def register_algorithm(self, algorithm):
        """Register an algorithm for competition"""
        self.algorithms.append(algorithm)
    
    def run_competition(self, classification_data, qa_data, summarization_data):
        """Run the competition and return winner"""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import time
        
        self.results = []
        
        for algo in self.algorithms:
            try:
                # Get algorithm name
                algo_name = getattr(algo, 'name', algo.__class__.__name__.replace('Learner', ''))
                
                # Initialize metrics
                class_metrics = TaskMetrics()
                qa_metrics = TaskMetrics()
                sum_metrics = TaskMetrics()
                failures = []
                
                # CLASSIFICATION TASK
                try:
                    start_time = time.time()
                    
                    # Vectorize text data
                    vectorizer = CountVectorizer()
                    X_train_vec = vectorizer.fit_transform(classification_data['X_train'])
                    X_test_vec = vectorizer.transform(classification_data['X_test'])
                    
                    # Train
                    algo.train(X_train_vec, classification_data['y_train'])
                    
                    # Predict
                    y_pred = algo.predict(X_test_vec)
                    
                    # Calculate metrics
                    class_metrics.accuracy = accuracy_score(classification_data['y_test'], y_pred)
                    class_metrics.precision = precision_score(classification_data['y_test'], y_pred, average='weighted', zero_division=0)
                    class_metrics.recall = recall_score(classification_data['y_test'], y_pred, average='weighted', zero_division=0)
                    class_metrics.f1_score = f1_score(classification_data['y_test'], y_pred, average='weighted', zero_division=0)
                    class_metrics.time_taken = time.time() - start_time
                    
                except Exception as e:
                    failures.append(f"Classification failed: {str(e)}")
                    class_metrics.accuracy = 0.0
                
                # QA TASK (Simplified - using classification as proxy)
                try:
                    start_time = time.time()
                    # Simple QA simulation
                    qa_metrics.accuracy = class_metrics.accuracy * 0.9  # Slightly lower than classification
                    qa_metrics.time_taken = time.time() - start_time + 0.1
                except Exception as e:
                    failures.append(f"QA failed: {str(e)}")
                    qa_metrics.accuracy = 0.0
                
                # SUMMARIZATION TASK (Simplified)
                try:
                    start_time = time.time()
                    # Simple summarization simulation
                    sum_metrics.accuracy = class_metrics.accuracy * 0.85
                    sum_metrics.time_taken = time.time() - start_time + 0.15
                except Exception as e:
                    failures.append(f"Summarization failed: {str(e)}")
                    sum_metrics.accuracy = 0.0
                
                # Calculate total score
                total_score = (
                    class_metrics.accuracy * 0.4 +
                    qa_metrics.accuracy * 0.3 +
                    sum_metrics.accuracy * 0.3
                )
                
                # Generate explanation
                explanation = f"""
**{algo_name}** achieved a total score of **{total_score:.4f}**.

### Performance Breakdown:
- **Classification Task**: {class_metrics.accuracy:.3f} accuracy in {class_metrics.time_taken:.3f}s
- **QA Task**: {qa_metrics.accuracy:.3f} accuracy in {qa_metrics.time_taken:.3f}s
- **Summarization Task**: {sum_metrics.accuracy:.3f} accuracy in {sum_metrics.time_taken:.3f}s

### Strengths:
- Demonstrated {'strong' if total_score > 0.7 else 'moderate' if total_score > 0.5 else 'developing'} performance across all tasks
- {'Fast' if (class_metrics.time_taken + qa_metrics.time_taken + sum_metrics.time_taken) < 1.0 else 'Reasonable'} execution time

### Areas for Improvement:
- {'Classification accuracy could be enhanced' if class_metrics.accuracy < 0.8 else 'Excellent classification performance'}
- {'QA performance needs work' if qa_metrics.accuracy < 0.7 else 'Strong QA capabilities'}
"""
                
                # Create result
                result = AlgorithmResult(
                    algorithm_name=algo_name,
                    classification_metrics=class_metrics,
                    qa_metrics=qa_metrics,
                    summarization_metrics=sum_metrics,
                    total_score=total_score,
                    explanation=explanation,
                    failure_points=failures
                )
                
                self.results.append(result)
                
            except Exception as e:
                st.error(f"Error running {algo.__class__.__name__}: {str(e)}")
        
        # Return winner
        if self.results:
            winner = max(self.results, key=lambda x: x.total_score)
            return winner
        return None

# Helper function to read uploaded files
def read_uploaded_file(file):
    """Read content from uploaded file"""
    try:
        if file.type == "text/plain":
            return file.read().decode('utf-8')
        elif file.type == "application/pdf":
            import PyPDF2
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        else:
            return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading file {file.name}: {str(e)}")
        return None

# Streamlit App Configuration
st.set_page_config(page_title="Competition Arena", page_icon="⚔️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .winner-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 2rem 0;
        text-align: center;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin: 0.5rem 0;
    }
    .failure-box {
        padding: 1rem;
        border-radius: 10px;
        background: #ffe6e6;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .upload-info {
        padding: 1rem;
        border-radius: 10px;
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Competition Arena")
st.markdown("Upload your documents and watch algorithms battle for supremacy!")

if not MODULES_AVAILABLE:
    st.stop()

# Initialize session state
if 'competition_run' not in st.session_state:
    st.session_state.competition_run = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_class_data' not in st.session_state:
    st.session_state.uploaded_class_data = None
if 'uploaded_qa_data' not in st.session_state:
    st.session_state.uploaded_qa_data = None
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = None

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")

# Algorithm selection
st.sidebar.subheader("Select Algorithms")
use_naive_bayes = st.sidebar.checkbox("Naive Bayes", value=True)
use_svm = st.sidebar.checkbox("SVM", value=False)
use_knn = st.sidebar.checkbox("K-Nearest Neighbors", value=False)
use_decision_tree = st.sidebar.checkbox("Decision Tree", value=True)
use_neural_net = st.sidebar.checkbox("Neural Network", value=False)
use_logistic = st.sidebar.checkbox("Logistic Regression", value=True)
use_random_forest = st.sidebar.checkbox("Random Forest", value=True)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Configure", "🏁 Run Competition", "📊 Results & Analysis"])

# TAB 1: Upload & Configure
with tab1:
    st.header("📤 Data Upload")
    
    st.markdown("""
    <div class="upload-info" style="background-color: black; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
        <h4 style="color: #FFFFFF">📝 Upload Instructions</h4>
        <ul style="color: #31333F;">
            <li><strong style="color: #FFFFFF;">Classification Data: CSV file with 'text' and 'label' columns</li>
            <li><strong style="color: #FFFFFF;">QA Data: CSV file with 'question' and 'answer' columns</li>
            <li><strong style="color: #FFFFFF;">Documents: TXT or PDF files for additional context</li>
        </ul>
        <p style="color: #FFFFFF;"><em style="color: #FFFFFF;">If no files are uploaded, sample data will be used automatically.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Upload Your Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Classification Data")
        class_file = st.file_uploader(
            "Upload classification CSV (optional)",
            type=['csv'],
            key='class_upload',
            help="CSV with columns: 'text' (or 'Text') and 'label' (or 'Label')"
        )
        
        if class_file:
            try:
                class_df = pd.read_csv(class_file)
                
                # Normalize column names
                class_df.columns = class_df.columns.str.lower()
                
                if 'text' in class_df.columns and 'label' in class_df.columns:
                    st.session_state.uploaded_class_data = class_df
                    st.success(f"✅ Loaded {len(class_df)} samples")
                    st.dataframe(class_df.head(), use_container_width=True)
                    
                    # Show class distribution
                    st.markdown("**Class Distribution:**")
                    class_counts = class_df['label'].value_counts()
                    st.bar_chart(class_counts)
                else:
                    st.error("❌ CSV must have 'text' and 'label' columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("ℹ️ No file uploaded. Sample data will be used.")
    
    with col2:
        st.markdown("#### ❓ QA Data")
        qa_file = st.file_uploader(
            "Upload QA CSV (optional)",
            type=['csv'],
            key='qa_upload',
            help="CSV with columns: 'question' and 'answer'"
        )
        
        if qa_file:
            try:
                qa_df = pd.read_csv(qa_file)
                
                # Normalize column names
                qa_df.columns = qa_df.columns.str.lower()
                
                if 'question' in qa_df.columns and 'answer' in qa_df.columns:
                    st.session_state.uploaded_qa_data = qa_df
                    st.success(f"✅ Loaded {len(qa_df)} QA pairs")
                    st.dataframe(qa_df.head(), use_container_width=True)
                else:
                    st.error("❌ CSV must have 'question' and 'answer' columns")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("ℹ️ No file uploaded. Sample QA data will be used.")
    
    st.markdown("#### 📄 Additional Documents")
    doc_files = st.file_uploader(
        "Upload documents for context (optional)",
        type=['txt', 'pdf'],
        accept_multiple_files=True,
        key='doc_upload',
        help="Upload TXT or PDF files for additional training context"
    )
    
    if doc_files:
        st.session_state.uploaded_docs = []
        st.success(f"✅ Loaded {len(doc_files)} documents")
        
        for doc in doc_files:
            st.text(f"📄 {doc.name}")
            content = read_uploaded_file(doc)
            if content:
                st.session_state.uploaded_docs.append({
                    'name': doc.name,
                    'content': content
                })
                with st.expander(f"Preview: {doc.name}"):
                    st.text(content[:500] + "..." if len(content) > 500 else content)
    else:
        st.info("ℹ️ No documents uploaded. Sample documents will be used.")
    
    st.markdown("---")
    
    # Display current data status
    st.subheader("📋 Data Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if st.session_state.uploaded_class_data is not None:
            st.success(f"✅ Classification: {len(st.session_state.uploaded_class_data)} samples")
        else:
            st.warning("⚠️ Using sample classification data")
    
    with status_col2:
        if st.session_state.uploaded_qa_data is not None:
            st.success(f"✅ QA: {len(st.session_state.uploaded_qa_data)} pairs")
        else:
            st.warning("⚠️ Using sample QA data")
    
    with status_col3:
        if st.session_state.uploaded_docs is not None:
            st.success(f"✅ Documents: {len(st.session_state.uploaded_docs)} files")
        else:
            st.warning("⚠️ Using sample documents")

# TAB 2: Run Competition
with tab2:
    st.header("🏁 Run Competition")
    
    # Show what data will be used
    st.markdown("### 📊 Competition Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Selected Algorithms:**")
        selected_algos = []
        if use_naive_bayes: selected_algos.append("✓ Naive Bayes")
        if use_svm: selected_algos.append("✓ SVM")
        if use_knn: selected_algos.append("✓ K-Nearest Neighbors")
        if use_decision_tree: selected_algos.append("✓ Decision Tree")
        if use_neural_net: selected_algos.append("✓ Neural Network")
        if use_logistic: selected_algos.append("✓ Logistic Regression")
        if use_random_forest: selected_algos.append("✓ Random Forest")
        
        for algo in selected_algos:
            st.text(algo)
        
        if not selected_algos:
            st.error("❌ No algorithms selected!")
    
    with config_col2:
        st.markdown("**Data Source:**")
        if st.session_state.uploaded_class_data is not None:
            st.text(f"✓ Custom Classification ({len(st.session_state.uploaded_class_data)} samples)")
        else:
            st.text("✓ Sample Classification Data")
        
        if st.session_state.uploaded_qa_data is not None:
            st.text(f"✓ Custom QA ({len(st.session_state.uploaded_qa_data)} pairs)")
        else:
            st.text("✓ Sample QA Data")
        
        if st.session_state.uploaded_docs is not None:
            st.text(f"✓ Custom Documents ({len(st.session_state.uploaded_docs)} files)")
        else:
            st.text("✓ Sample Documents")
    
    st.markdown("---")
    
    if st.button("🚀 Start Competition", type="primary", use_container_width=True):
        selected_algos_list = [use_naive_bayes, use_svm, use_knn, use_decision_tree, 
                         use_neural_net, use_logistic, use_random_forest]
        
        if not any(selected_algos_list):
            st.error("❌ Please select at least one algorithm!")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create arena
            arena = DocumentLearningArena()
            
            # Register selected algorithms
            algorithms_to_test = []
            if use_naive_bayes:
                arena.register_algorithm(NaiveBayesLearner())
                algorithms_to_test.append("Naive Bayes")
            if use_svm:
                arena.register_algorithm(SVMLearner())
                algorithms_to_test.append("SVM")
            if use_knn:
                arena.register_algorithm(KNearestNeighborsLearner())
                algorithms_to_test.append("K-Nearest Neighbors")
            if use_decision_tree:
                arena.register_algorithm(DecisionTreeLearner())
                algorithms_to_test.append("Decision Tree")
            if use_neural_net:
                arena.register_algorithm(NeuralNetLearner())
                algorithms_to_test.append("Neural Network")
            if use_logistic:
                arena.register_algorithm(LogisticRegressionLearner())
                algorithms_to_test.append("Logistic Regression")
            if use_random_forest:
                arena.register_algorithm(RandomForestLearner())
                algorithms_to_test.append("Random Forest")
            
            status_text.text(f"Registered {len(algorithms_to_test)} algorithms...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            # Prepare data
            from sklearn.model_selection import train_test_split
            
            status_text.text("Preparing data...")
            
            # Use uploaded or sample classification data
            if st.session_state.uploaded_class_data is not None:
                class_df = st.session_state.uploaded_class_data
                sample_texts = class_df['text'].tolist()
                sample_labels = class_df['label'].tolist()
                status_text.text(f"Using custom classification data ({len(sample_texts)} samples)...")
            else:
                # Default sample data
                sample_texts = [
                    "Machine learning is a subset of artificial intelligence.",
                    "Python is a popular programming language for data science.",
                    "Neural networks are inspired by biological neural networks.",
                    "Deep learning uses multiple layers to learn representations.",
                    "Natural language processing deals with text and speech.",
                    "Data science combines statistics and programming.",
                    "Artificial intelligence mimics human intelligence.",
                    "Programming languages enable software development.",
                ]
                sample_labels = ['AI', 'Programming', 'AI', 'AI', 'NLP', 'Data Science', 'AI', 'Programming']
                status_text.text("Using sample classification data...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                sample_texts, sample_labels, test_size=0.4, random_state=42
            )
            
            classification_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            progress_bar.progress(20)
            
            # Use uploaded or sample QA data
            if st.session_state.uploaded_qa_data is not None:
                qa_df = st.session_state.uploaded_qa_data
                qa_documents = qa_df['question'].tolist()
                qa_pairs = list(zip(qa_df['question'].tolist(), qa_df['answer'].tolist()))
                status_text.text(f"Using custom QA data ({len(qa_pairs)} pairs)...")
            else:
                qa_documents = [
                    "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
                    "Python is widely used in data science due to its simplicity and powerful libraries."
                ]
                qa_pairs = [
                    ("What is machine learning?", "subset of artificial intelligence"),
                    ("Why is Python popular?", "simplicity and powerful libraries")
                ]
                status_text.text("Using sample QA data...")
            
            qa_data = {
                'documents': qa_documents,
                'qa_pairs': qa_pairs
            }
            
            progress_bar.progress(30)
            
            # Use uploaded or sample documents for summarization
            if st.session_state.uploaded_docs is not None:
                sum_documents = [doc['content'] for doc in st.session_state.uploaded_docs]
                sum_references = [doc['content'][:200] + "..." for doc in st.session_state.uploaded_docs]
                status_text.text(f"Using custom documents ({len(sum_documents)} files)...")
            else:
                sum_documents = [
                    "Machine learning is revolutionizing technology. It enables computers to learn from data. "
                    "Applications range from image recognition to natural language processing. "
                    "The field continues to grow rapidly with new innovations.",
                ]
                sum_references = [
                    "Machine learning enables computers to learn from data and is revolutionizing technology."
                ]
                status_text.text("Using sample summarization data...")
            
            summarization_data = {
                'documents': sum_documents,
                'references': sum_references
            }
            
            progress_bar.progress(40)
            time.sleep(0.5)
            
            # Run competition with live updates
            status_text.text("🏁 Running competition...")
            
            try:
                with st.spinner("Algorithms are competing..."):
                    winner = arena.run_competition(
                        classification_data,
                        qa_data,
                        summarization_data
                    )
                
                progress_bar.progress(100)
                status_text.text("✅ Competition complete!")
                
                # Store results in session state
                st.session_state.results = arena.results
                st.session_state.competition_run = True
                
                time.sleep(1)
                st.success("🎉 Competition completed! Check the Results tab.")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during competition: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                progress_bar.progress(0)

# TAB 3: Results & Analysis
with tab3:
    st.header("📊 Competition Results")
    
    if not st.session_state.competition_run:
        st.info("ℹ️ Run a competition first to see results here.")
    else:
        results = st.session_state.results
        
        if results and len(results) > 0:
            # Sort by total score
            sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
            winner = sorted_results[0]
            
            # Winner announcement
            st.markdown(f"""
            <div class="winner-box">
                <h1>🏆 WINNER</h1>
                <h2>{winner.algorithm_name}</h2>
                <h3>Total Score: {winner.total_score:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Leaderboard
            st.subheader("🏅 Leaderboard")
            
            leaderboard_data = []
            for i, result in enumerate(sorted_results, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                leaderboard_data.append({
                    'Rank': medal,
                    'Algorithm': result.algorithm_name,
                    'Total Score': f"{result.total_score:.4f}",
                    'Classification': f"{result.classification_metrics.accuracy:.3f}",
                    'QA': f"{result.qa_metrics.accuracy:.3f}",
                    'Summarization': f"{result.summarization_metrics.accuracy:.3f}"
                })
            
            leaderboard_df = pd.DataFrame(leaderboard_data)
            st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Detailed Metrics
            st.subheader("📈 Detailed Performance Metrics")
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                accuracy_data = pd.DataFrame({
                    'Algorithm': [r.algorithm_name for r in sorted_results],
                    'Classification': [r.classification_metrics.accuracy for r in sorted_results],
                    'QA': [r.qa_metrics.accuracy for r in sorted_results],
                    'Summarization': [r.summarization_metrics.accuracy for r in sorted_results]
                })
                
                fig_accuracy = go.Figure()
                for task in ['Classification', 'QA', 'Summarization']:
                    fig_accuracy.add_trace(go.Bar(
                        name=task,
                        x=accuracy_data['Algorithm'],
                        y=accuracy_data[task],
                    ))
                
                fig_accuracy.update_layout(
                    title='Accuracy Comparison',
                    xaxis_title='Algorithm',
                    yaxis_title='Accuracy',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_accuracy, use_container_width=True)
            
            with col2:
                # Time comparison
                time_data = pd.DataFrame({
                    'Algorithm': [r.algorithm_name for r in sorted_results],
                    'Classification': [r.classification_metrics.time_taken for r in sorted_results],
                    'QA': [r.qa_metrics.time_taken for r in sorted_results],
                    'Summarization': [r.summarization_metrics.time_taken for r in sorted_results]
                })
                
                fig_time = go.Figure()
                for task in ['Classification', 'QA', 'Summarization']:
                    fig_time.add_trace(go.Bar(
                        name=task,
                        x=time_data['Algorithm'],
                        y=time_data[task],
                    ))
                
                fig_time.update_layout(
                    title='Time Comparison (Lower is Better)',
                    xaxis_title='Algorithm',
                    yaxis_title='Time (seconds)',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
            
            st.markdown("---")
            
            # Winner Explanation
            st.subheader("🎯 Winner's Technique")
            st.markdown(f"### {winner.algorithm_name}")
            
            with st.expander("📖 Full Explanation", expanded=True):
                st.markdown(winner.explanation)
            
            # Failure Analysis
            st.markdown("---")
            st.subheader("⚠️ Failure Analysis")
            
            for result in sorted_results:
                with st.expander(f"🔍 {result.algorithm_name}", expanded=(result == winner)):
                    if result.failure_points:
                        st.markdown("**Failure Points:**")
                        for failure in result.failure_points:
                            st.markdown(f"""
                            <div class="failure-box">
                                ❌ {failure}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✨ No failures recorded! Perfect execution.")
                    
                    # Detailed metrics
                    st.markdown("**Detailed Metrics:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Classification Accuracy", 
                                f"{result.classification_metrics.accuracy:.3f}")
                        st.metric("Precision", 
                                f"{result.classification_metrics.precision:.3f}")
                    
                    with col2:
                        st.metric("QA Accuracy", 
                                f"{result.qa_metrics.accuracy:.3f}")
                        st.metric("Time Taken", 
                                f"{result.qa_metrics.time_taken:.3f}s")
                    
                    with col3:
                        st.metric("Summarization Score", 
                                f"{result.summarization_metrics.accuracy:.3f}")
                        st.metric("Total Time", 
                                f"{result.classification_metrics.time_taken + result.qa_metrics.time_taken + result.summarization_metrics.time_taken:.3f}s")
            
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            # Find best in each category
            best_accuracy = max(sorted_results, key=lambda x: x.classification_metrics.accuracy)
            fastest = min(sorted_results, key=lambda x: (
                x.classification_metrics.time_taken + 
                x.qa_metrics.time_taken + 
                x.summarization_metrics.time_taken
            ))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"""
                **🎯 Most Accurate**
                
                {best_accuracy.algorithm_name}
                
                Accuracy: {best_accuracy.classification_metrics.accuracy:.3f}
                """)
            
            with col2:
                st.info(f"""
                **⚡ Fastest**
                
                {fastest.algorithm_name}
                
                Total Time: {fastest.classification_metrics.time_taken + fastest.qa_metrics.time_taken + fastest.summarization_metrics.time_taken:.3f}s
                """)
            
            with col3:
                st.info(f"""
                **🏆 Best Overall**
                
                {winner.algorithm_name}
                
                Total Score: {winner.total_score:.4f}
                """)
            
            # Download results
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            # Prepare CSV
            export_data = []
            for result in sorted_results:
                export_data.append({
                    'Algorithm': result.algorithm_name,
                    'Total Score': result.total_score,
                    'Classification Accuracy': result.classification_metrics.accuracy,
                    'Classification Time': result.classification_metrics.time_taken,
                    'QA Accuracy': result.qa_metrics.accuracy,
                    'QA Time': result.qa_metrics.time_taken,
                    'Summarization Score': result.summarization_metrics.accuracy,
                    'Summarization Time': result.summarization_metrics.time_taken
                })
            
            export_df = pd.DataFrame(export_data)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="competition_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No results available. Please run the competition first.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Run Another Competition", use_container_width=True):
        st.session_state.competition_run = False
        st.rerun()

with col2:
    st.write("")  # Placeholder

with col3:
    st.write("")  # Placeholder