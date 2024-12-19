import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from deep_translator import GoogleTranslator

# Custom CSS to change background color and title color
st.markdown("""
    <style>
    .main {
        background-color: white;
    }
    .title {
        color: #003990;
    }
    </style>
""", unsafe_allow_html=True)

# Load standard responsibilities and their mapping to HR pillars
standard_responsibilities = pd.DataFrame({
    'Standard Responsibility': [
        "Formulation of Organizational Change Process",
        "Organizational Diagnosis",
        "Design of Organizational Change Plan"],
    'HRCOE Score': [
        1.44, 0.72, 0.72],
    'HRBP Score': [
        0, 1.44, 1.44],
    'HRSSC Score': [
        0, 0, 0]
})

# Streamlit app configuration
st.title("TDG HROT Tool")

# Step 1: User inputs employee responsibilities in Chinese
st.header("Please Input Your Responsibilities (in Chinese)")
input_responsibilities = st.text_area(
    "Enter responsibilities (one per line):", 
    "")

if st.button("Analyze Responsibilities"):
    if input_responsibilities.strip():
        # Translate the Chinese responsibilities to English using deep_translator
        translated_responsibilities = [GoogleTranslator(source='zh-CN', target='en').translate(resp.strip()) for resp in input_responsibilities.splitlines() if resp.strip()]
        
        # Step 2: Analyze and map responsibilities
        st.header("Mapped Responsibilities")
        
        # Vectorization for similarity analysis
        tfidf = TfidfVectorizer()
        all_texts = standard_responsibilities['Standard Responsibility'].tolist() + translated_responsibilities
        tfidf_matrix = tfidf.fit_transform(all_texts)
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(
            tfidf_matrix[len(standard_responsibilities):],
            tfidf_matrix[:len(standard_responsibilities)]
        )
        
        # Mapping responsibilities to closest standard responsibility
        mapped_results = []
        pillar_scores = {'HRCOE': 0, 'HRBP': 0, 'HRSSC': 0}
        
        for i, emp_resp in enumerate(translated_responsibilities):
            max_index = similarity_matrix[i].argmax()
            standard_resp = standard_responsibilities.iloc[max_index]
            mapped_results.append({
                'Employee Responsibility': emp_resp,
                'Mapped Standard Responsibility': standard_resp['Standard Responsibility']
            })
            
            # Accumulate pillar scores
            pillar_scores['HRCOE'] += standard_resp['HRCOE Score']
            pillar_scores['HRBP'] += standard_resp['HRBP Score']
            pillar_scores['HRSSC'] += standard_resp['HRSSC Score']
        
        # Display mapped results without scores
        results_df = pd.DataFrame(mapped_results)
        st.write(results_df)
        
        # Step 3: Calculate percentages for HR pillars
        total_score = sum(pillar_scores.values())
        if total_score > 0:
            for pillar in pillar_scores:
                pillar_scores[pillar] = (pillar_scores[pillar] / total_score) * 100
        
        # Create a donut chart for pillar scores
        st.header("Pillar Scores (as percentages)")
        fig = px.pie(
            values=list(pillar_scores.values()),
            names=list(pillar_scores.keys()),
            title="Pillar Scores Distribution",
            hole=0.5  # Make it a donut chart
        )

        # Adjust the chart size
        fig.update_layout(
            width=500,  # Set chart width
            height=500,  # Set chart height
            margin=dict(t=40, b=40, l=40, r=40)  # Add some margin around the chart
        )

        st.plotly_chart(fig)
        
        # Step 4: Determine the dominant HR pillar
        dominant_pillar = max(pillar_scores, key=pillar_scores.get)
        st.success(f"This employee aligns most with the {dominant_pillar} pillar.")
  
