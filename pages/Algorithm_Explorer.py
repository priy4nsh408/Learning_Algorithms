import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from competition_module import (
    NaiveBayesLearner,
    DecisionTreeLearner,
    NeuralNetLearner,
    KNearestNeighborsLearner,
    SVMLearner,
    LogisticRegressionLearner,
    RandomForestLearner

)

st.set_page_config(
    page_title="Algorithm Explorer",
    page_icon="📚",
    layout="wide"
)

# Algorithm information database
ALGORITHMS = {
    "Naive Bayes": {
        "class": NaiveBayesLearner,
        "description": "Probabilistic classifier based on Bayes' theorem with strong independence assumptions.",
        "strengths": [
            "Fast training and prediction",
            "Works well with small datasets",
            "Handles high-dimensional data",
            "Good for text classification"
        ],
        "weaknesses": [
            "Assumes feature independence",
            "Sensitive to irrelevant features",
            "Zero-frequency problem",
            "Poor probability estimates"
        ],
        "use_cases": [
            "Spam filtering",
            "Document classification",
            "Sentiment analysis",
            "Real-time prediction"
        ],
        "complexity": {
            "time_train": "O(n × d)",
            "time_predict": "O(d × k)",
            "space": "O(d × k)"
        },
        "parameters": {
            "smoothing": "Laplace smoothing factor (default: 1.0)"
        }
    },
    "Decision Tree": {
        "class": DecisionTreeLearner,
        "description": "Tree-based model that makes decisions by learning simple decision rules from features.",
        "strengths": [
            "Easy to interpret and visualize",
            "Handles non-linear relationships",
            "No feature scaling needed",
            "Captures feature interactions"
        ],
        "weaknesses": [
            "Prone to overfitting",
            "Unstable with small data changes",
            "Biased with imbalanced data",
            "Can create complex trees"
        ],
        "use_cases": [
            "Medical diagnosis",
            "Credit risk assessment",
            "Customer segmentation",
            "Rule extraction"
        ],
        "complexity": {
            "time_train": "O(n × d × log(n))",
            "time_predict": "O(log(n))",
            "space": "O(nodes)"
        },
        "parameters": {
            "max_depth": "Maximum tree depth",
            "min_samples_split": "Minimum samples to split node",
            "criterion": "Split quality measure (gini/entropy)"
        }
    },
    "Neural Network": {
        "class": NeuralNetLearner,
        "description": "Multi-layer perceptron with backpropagation learning for complex pattern recognition.",
        "strengths": [
            "Learns complex patterns",
            "Handles non-linear data",
            "Flexible architecture",
            "Good with large datasets"
        ],
        "weaknesses": [
            "Requires large data",
            "Black box model",
            "Computationally expensive",
            "Sensitive to hyperparameters"
        ],
        "use_cases": [
            "Image recognition",
            "Speech recognition",
            "Language modeling",
            "Game playing"
        ],
        "complexity": {
            "time_train": "O(epochs × n × weights)",
            "time_predict": "O(layers × nodes)",
            "space": "O(weights + activations)"
        },
        "parameters": {
            "hidden_layers": "Network architecture",
            "learning_rate": "Step size for updates",
            "epochs": "Training iterations"
        }
    },
    "K-Nearest Neighbors": {
        "class": KNearestNeighborsLearner,
        "description": "Instance-based learning that classifies based on closest training examples.",
        "strengths": [
            "No training phase",
            "Simple and intuitive",
            "Naturally handles multi-class",
            "Non-parametric"
        ],
        "weaknesses": [
            "Slow prediction time",
            "Sensitive to irrelevant features",
            "Requires feature scaling",
            "Memory intensive"
        ],
        "use_cases": [
            "Recommendation systems",
            "Pattern recognition",
            "Anomaly detection",
            "Missing value imputation"
        ],
        "complexity": {
            "time_train": "O(1)",
            "time_predict": "O(n × d)",
            "space": "O(n × d)"
        },
        "parameters": {
            "k": "Number of neighbors",
            "distance_metric": "Similarity measure",
            "weighting": "Distance weighting scheme"
        }
    },
    "Support Vector Machine": {
        "class": SVMLearner,
        "description": "Finds optimal hyperplane that maximizes margin between classes.",
        "strengths": [
            "Effective in high dimensions",
            "Memory efficient",
            "Versatile with kernels",
            "Good with clear margin"
        ],
        "weaknesses": [
            "Slow with large datasets",
            "Sensitive to noise",
            "Requires feature scaling",
            "Black box with kernels"
        ],
        "use_cases": [
            "Image classification",
            "Text categorization",
            "Bioinformatics",
            "Handwriting recognition"
        ],
        "complexity": {
            "time_train": "O(n² × d) to O(n³ × d)",
            "time_predict": "O(n_sv × d)",
            "space": "O(n_sv × d)"
        },
        "parameters": {
            "C": "Regularization parameter",
            "kernel": "Kernel function type",
            "gamma": "Kernel coefficient"
        }
    },
     "Logistic Regression": {
     "class": LogisticRegressionLearner,
     "description": "Linear classification model that estimates the probability of a binary outcome using the logistic (sigmoid) function.",
     "strengths": [
        "Simple and fast to train",
        "Works well with linearly separable data",
        "Outputs probabilistic predictions",
        "Less prone to overfitting with regularization"
    ],
    "weaknesses": [
        "Cannot model complex non-linear relationships",
        "Sensitive to outliers",
        "Requires feature scaling",
        "Assumes linear decision boundary"
    ],
    "use_cases": [
        "Spam detection",
        "Disease prediction",
        "Credit scoring",
        "Binary classification problems"
    ],
    "complexity": {
        "time_train": "O(n × d)",
        "time_predict": "O(d)",
        "space": "O(d)"
    },
    "parameters": {
        "penalty": "Regularization type (l1, l2)",
        "C": "Inverse of regularization strength",
        "solver": "Optimization algorithm (liblinear, lbfgs)"
    }
},
"Random Forest": {
    "class": RandomForestLearner,
    "description": "Ensemble learning method that builds multiple decision trees and aggregates their predictions for better accuracy and robustness.",
    "strengths": [
        "Reduces overfitting compared to single trees",
        "Handles non-linear data well",
        "Works with high-dimensional data",
        "Robust to noise and outliers"
    ],
    "weaknesses": [
        "Less interpretable than a single tree",
        "Higher computational cost",
        "Large memory usage",
        "Slower prediction with many trees"
    ],
    "use_cases": [
        "Fraud detection",
        "Recommendation systems",
        "Medical diagnosis",
        "Feature importance analysis"
    ],
    "complexity": {
        "time_train": "O(k × n × d × log(n))",
        "time_predict": "O(k × log(n))",
        "space": "O(k × nodes)"
    },
    "parameters": {
        "n_estimators": "Number of trees in the forest",
        "max_depth": "Maximum depth of each tree",
        "max_features": "Number of features to consider at each split"
    }
}


}

# Header
st.title("📚 Algorithm Explorer")
st.markdown("**Discover and compare machine learning algorithms**")

# Sidebar for algorithm selection
st.sidebar.header("Select Algorithms")
selected_algos = st.sidebar.multiselect(
    "Choose algorithms to explore:",
    list(ALGORITHMS.keys()),
    default=["Naive Bayes", "Decision Tree"]
)

if not selected_algos:
    st.info("👈 Select at least one algorithm from the sidebar to begin exploration")
    st.stop()

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "⚡ Comparison", "📊 Complexity", "🎯 Use Cases"])

# Tab 1: Overview
with tab1:
    for algo_name in selected_algos:
        algo_info = ALGORITHMS[algo_name]
        
        with st.expander(f"**{algo_name}**", expanded=True):
            st.markdown(f"### {algo_name}")
            st.markdown(algo_info["description"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Strengths**")
                for strength in algo_info["strengths"]:
                    st.markdown(f"- {strength}")
            
            with col2:
                st.markdown("**⚠️ Weaknesses**")
                for weakness in algo_info["weaknesses"]:
                    st.markdown(f"- {weakness}")
            
            st.markdown("**⚙️ Key Parameters**")
            for param, desc in algo_info["parameters"].items():
                st.markdown(f"- **{param}:** {desc}")

# Tab 2: Comparison
with tab2:
    st.header("Algorithm Comparison")
    
    # Create comparison metrics
    comparison_data = []
    for algo_name in selected_algos:
        algo_info = ALGORITHMS[algo_name]
        comparison_data.append({
            "Algorithm": algo_name,
            "Training Speed": "Fast" if "Fast" in str(algo_info["strengths"]) else "Moderate" if "n²" not in algo_info["complexity"]["time_train"] else "Slow",
            "Prediction Speed": "Fast" if "log(n)" in algo_info["complexity"]["time_predict"] or "d" in algo_info["complexity"]["time_predict"] else "Moderate",
            "Interpretability": "High" if algo_name in ["Decision Tree", "Naive Bayes"] else "Medium" if algo_name == "K-Nearest Neighbors" else "Low",
            "Scalability": "High" if algo_name in ["Naive Bayes"] else "Medium" if algo_name in ["Decision Tree", "Neural Network"] else "Low",
            "Strengths": len(algo_info["strengths"]),
            "Weaknesses": len(algo_info["weaknesses"])
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Radar chart
    categories = ["Training Speed", "Prediction Speed", "Interpretability", "Scalability"]
    
    # Convert to numeric scores
    speed_map = {"Fast": 3, "Moderate": 2, "Slow": 1}
    interpret_map = {"High": 3, "Medium": 2, "Low": 1}
    
    fig_radar = go.Figure()
    
    for algo_name in selected_algos:
        row = df_comparison[df_comparison["Algorithm"] == algo_name].iloc[0]
        values = [
            speed_map[row["Training Speed"]],
            speed_map[row["Prediction Speed"]],
            interpret_map[row["Interpretability"]],
            interpret_map[row["Scalability"]]
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=algo_name
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 3])),
        showlegend=True,
        height=500,
        title="Algorithm Performance Comparison"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Comparison table
    st.markdown("### Detailed Comparison")
    st.dataframe(df_comparison.set_index("Algorithm"), use_container_width=True)

# Tab 3: Complexity Analysis
with tab3:
    st.header("Computational Complexity")
    
    for algo_name in selected_algos:
        algo_info = ALGORITHMS[algo_name]
        
        with st.expander(f"**{algo_name} Complexity**", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Training Time", algo_info["complexity"]["time_train"])
            
            with col2:
                st.metric("Prediction Time", algo_info["complexity"]["time_predict"])
            
            with col3:
                st.metric("Space", algo_info["complexity"]["space"])
            
            st.markdown("**Notation:**")
            st.markdown("- **n:** Number of training examples")
            st.markdown("- **d:** Number of features/dimensions")
            st.markdown("- **k:** Number of classes")
            st.markdown("- **n_sv:** Number of support vectors")
    
    # Complexity visualization
    st.markdown("### Training Time Growth")
    
    # Generate sample data
    n_samples = list(range(100, 10001, 500))
    
    complexity_data = {
        "Samples": n_samples,
        "Naive Bayes": [n * 100 for n in n_samples],  # O(n×d)
        "Decision Tree": [n * 100 * (n ** 0.5) for n in n_samples],  # O(n×d×log(n))
        "K-NN": [1] * len(n_samples),  # O(1)
        "Neural Network": [n * 1000 for n in n_samples],  # O(epochs×n×weights)
        "SVM": [n ** 2 * 0.1 for n in n_samples]  # O(n²×d)
    }
    
    df_complexity = pd.DataFrame(complexity_data)
    
    # Filter for selected algorithms
    cols_to_plot = ["Samples"] + [algo for algo in selected_algos if algo in df_complexity.columns]
    df_plot = df_complexity[cols_to_plot]
    
    fig_complexity = px.line(
        df_plot,
        x="Samples",
        y=df_plot.columns[1:],
        title="Training Time vs Dataset Size (Relative)",
        labels={"value": "Relative Time", "variable": "Algorithm"}
    )
    
    fig_complexity.update_layout(height=500)
    st.plotly_chart(fig_complexity, use_container_width=True)

# Tab 4: Use Cases
with tab4:
    st.header("Real-World Applications")
    
    for algo_name in selected_algos:
        algo_info = ALGORITHMS[algo_name]
        
        st.markdown(f"### {algo_name}")
        
        cols = st.columns(2)
        for idx, use_case in enumerate(algo_info["use_cases"]):
            with cols[idx % 2]:
                st.markdown(f"**{idx + 1}. {use_case}**")
                
                # Add context based on use case
                if "classification" in use_case.lower() or "categorization" in use_case.lower():
                    st.caption("Assigning items to predefined categories")
                elif "recognition" in use_case.lower():
                    st.caption("Identifying patterns in data")
                elif "recommendation" in use_case.lower():
                    st.caption("Suggesting items based on similarity")
                elif "detection" in use_case.lower():
                    st.caption("Finding unusual patterns or outliers")
                elif "diagnosis" in use_case.lower():
                    st.caption("Making decisions based on symptoms/features")
                else:
                    st.caption("Solving complex real-world problems")
        
        st.divider()

# Footer
st.markdown("---")
st.markdown("**💡 Tip:** Use the Competition Arena to see these algorithms in action on real data!")