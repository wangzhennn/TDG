import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Step 1: User inputs employee responsibilities
st.header("Please Input Your Responsibilities")
input_responsibilities = st.text_area(
    "Enter responsibilities (one per line):", 
    "")

if st.button("Analyze Responsibilities"):
    if input_responsibilities.strip():
        employee_responsibilities = [resp.strip() for resp in input_responsibilities.splitlines() if resp.strip()]
        st.write("## Entered Responsibilities:")
        st.write(employee_responsibilities)
        
        # Step 2: Analyze and map responsibilities
        st.header("Mapped Responsibilities")
        
        # Vectorization for similarity analysis
        tfidf = TfidfVectorizer()
        all_texts = standard_responsibilities['Standard Responsibility'].tolist() + employee_responsibilities
        tfidf_matrix = tfidf.fit_transform(all_texts)
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(
            tfidf_matrix[len(standard_responsibilities):],
            tfidf_matrix[:len(standard_responsibilities)]
        )
        
        # Mapping responsibilities to closest standard responsibility
        mapped_results = []
        pillar_scores = {'HRCOE': 0, 'HRBP': 0, 'HRSSC': 0}
        
        for i, emp_resp in enumerate(employee_responsibilities):
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
        
        st.header("Pillar Scores (as percentages)")
        st.write(pillar_scores)
        
        # Step 4: Determine the dominant HR pillar
        dominant_pillar = max(pillar_scores, key=pillar_scores.get)
        st.success(f"This employee aligns most with the {dominant_pillar} pillar.")
    else:
        st.warning("Please input at least one responsibility.")
