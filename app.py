import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Competitive Document Learning",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .feature-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Main page content
def main():
    st.markdown('<h1 class="main-header">🏆 Competitive Document Learning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Where Algorithms Battle for Document Understanding Supremacy</p>', unsafe_allow_html=True)
    
    # Introduction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/600x300/667eea/ffffff?text=ML+Competition+Arena", use_container_width=True)
    
    st.markdown("---")
    
    # Features
    st.markdown("## 🎯 What Can You Do?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📚 Learn Algorithms</h3>
            <p>Explore built-in ML algorithms including Naive Bayes, Logistic Regression, and Random Forest. Understand their strengths and weaknesses.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>⚔️ Watch Them Compete</h3>
            <p>Upload your documents and watch algorithms compete in real-time on Classification, QA, and Summarization tasks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>💻 Build Your Own</h3>
            <p>Use the integrated Python compiler to create and test your own custom learning algorithms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("## 📊 Platform Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Algorithms", "3+", "Built-in")
    
    with col2:
        st.metric("Tasks", "3", "Multi-task evaluation")
    
    with col3:
        st.metric("Metrics", "5+", "Performance tracking")
    
    with col4:
        st.metric("Code Editor", "1", "Integrated")
    
    st.markdown("---")
    
    # How it works
    st.markdown("## 🔄 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ **Choose Your Path**
        - Learn about existing algorithms
        - Test algorithms on your data
        - Build custom solutions
        
        ### 2️⃣ **Upload Documents**
        - Text files (.txt)
        - PDF documents (.pdf)
        - Word documents (.docx)
        """)
    
    with col2:
        st.markdown("""
        ### 3️⃣ **Watch Competition**
        - Real-time performance metrics
        - Accuracy and speed tracking
        - Winner explanation
        
        ### 4️⃣ **Learn & Improve**
        - Understand failure points
        - Compare techniques
        - Iterate on your approach
        """)
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("## 🚀 Get Started")
    
    st.info("👈 Use the sidebar to navigate between different pages of the application.")
    
    # Check which pages exist
    pages_dir = Path("pages")
    page_files = {}
    
    if pages_dir.exists():
        for page_file in pages_dir.glob("*.py"):
            page_files[page_file.stem] = str(page_file)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Explore Algorithms", use_container_width=True):
            # Try different possible names
            possible_names = [
                "pages/1_📚_Algorithm_Explorer.py",
                "pages/Algorithm_Explorer.py",
                "pages/1_Algorithm_Explorer.py"
            ]
            for page_path in possible_names:
                if Path(page_path).exists():
                    st.switch_page(page_path)
                    break
            else:
                st.error("⚠️ Algorithm Explorer page not found. Please create: `pages/1_📚_Algorithm_Explorer.py`")
    
    with col2:
        if st.button("⚔️ Run Competition", use_container_width=True):
            possible_names = [
                "pages/2_⚔️_Competition_Arena.py",
                "pages/Competition_Arena.py",
                "pages/2_Competition_Arena.py"
            ]
            for page_path in possible_names:
                if Path(page_path).exists():
                    st.switch_page(page_path)
                    break
            else:
                st.error("⚠️ Competition Arena page not found. Please create: `pages/2_⚔️_Competition_Arena.py`")
    
    with col3:
        if st.button("💻 Code Editor", use_container_width=True):
            possible_names = [
                "pages/3_💻_Algorithm_Builder.py",
                "pages/Algorithm_Builder.py",
                "pages/3_Algorithm_Builder.py"
            ]
            for page_path in possible_names:
                if Path(page_path).exists():
                    st.switch_page(page_path)
                    break
            else:
                st.error("⚠️ Algorithm Builder page not found. Please create: `pages/3_💻_Algorithm_Builder.py`")
    
    st.markdown("---")

 
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with ❤️ using Streamlit and scikit-learn</p>
        <p>🔗 <a href='https://github.com/yourusername/compete-document'>GitHub Repository</a> | 📖 <a href='#'>Documentation</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()