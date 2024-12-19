import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
from deep_translator import GoogleTranslator


# Load standard responsibilities and their mapping to HR pillars
standard_responsibilities = pd.DataFrame({
    'Standard Responsibility': [
        'Formulation of Organizational Change Process', 
        'Organizational Diagnosis', 
        'Design of Organizational Change Plan', 
        'Release of Organizational Change Plan', 
        'Organizational Culture Design', 
        'Organizational Culture Implementation and Change Design', 
        'Design of Organizational Culture Activities', 
        'Implementation of Organizational Culture Activities', 
        'Design/Modification of Job Grading System', 
        'Development/Maintenance of Job Setting Standards', 
        'Job Setting/Modification', 
        'Job Value Assessment', 
        'Development of Overall HR Plan (including TianTong Innovation Academy)', 
        'Efficiency and Headcount Management', 
        'HR Budget Management', 
        'HR Cost Management', 
        'HR Policy Management', 
        'Pre-Investment Due Diligence and Post-Investment Management', 
        'Development of Shared HR Service Policies', 
        'Change Management Support', 
        'HR Digital Strategy Management', 
        'Development/Maintenance of Organizational Performance Management System', 
        'Organizational Performance Target Setting/Modification', 
        'Organizational Performance Evaluation', 
        'Organizational Performance Management Appeals Handling', 
        'Development/Maintenance of Recruitment and Allocation Policies', 
        'Recruitment Planning', 
        'Recruitment Demand Management', 
        'External Recruitment Channel Management', 
        'External Recruitment Implementation', 
        'Employer Branding', 
        'Mandatory Allocation', 
        'Internal Talent Market Allocation', 
        'Long-Term Assignment Arrangements', 
        'Labor Dispatch Demand Management', 
        'Labor Dispatch Implementation', 
        'Probation Management', 
        'Employee Resignation Management', 
        'Employee Retirement Management', 
        'Management of Underperformers’ Exit', 
        'Labor Contract Renewal Management', 
        'Management of Qualification Standards', 
        'Qualification Certification', 
        'Development of High-Potential Talent Standards', 
        'High-Potential Talent Identification Scheme Design', 
        'Implementation of High-Potential Talent Identification', 
        'Development/Maintenance of Career Development Policies', 
        'Implementation of Talent Promotions, Demotions, and Transfers', 
        'Planning for Talent Capability Development', 
        'Planning for On-Job Talent Capability Enhancement', 
        'Development/Maintenance of Performance Management Policies', 
        'Setting/Modification of Individual Performance Goals', 
        'Individual Performance Evaluation', 
        'Guidance on Individual Performance Feedback', 
        'Handling of Individual Performance Management Appeals', 
        'Development/Maintenance of Compensation and Benefits Systems', 
        'Salary Market Research and Analysis', 
        'Salary Adjustment Management', 
        'Design of Bonus Incentive Plans', 
        'Bonus Evaluation', 
        'Bonus Appeals', 
        'Design of Long-Term Incentive Plans', 
        'Allocation of Long-Term Incentives', 
        'Management of Social Security Rules and Schemes', 
        'Management of Social Security for Chinese Employees', 
        'Management of Subsidiaries’ Social Security and Housing Fund Accounts', 
        'Management of Non-Security Benefits', 
        'Management of Tax Rules and Schemes', 
        'Tax Calculation', 
        'Tax Risk Management', 
        'Development/Maintenance of Cadre Management Policies', 
        'Development of Cadre Standards', 
        'Cadre Planning', 
        'Cadre Selection and Appointment', 
        'Development of Cadre Training Plans', 
        'Cadre Term Evaluation', 
        'Cadre Allocation Planning', 
        'Implementation of Cadre Allocation', 
        'Cadre Exit Management', 
        'Cadre Investigation', 
        'Design of Annual Cadre Review Plan', 
        'Implementation of Annual Cadre Review Plan', 
        'Development of Cadre Succession Plans and Schemes', 
        'Implementation of Cadre Succession Plans and Schemes', 
        'Design of Honor Incentive Framework', 
        'Organization of Honor Award Evaluations', 
        'Employee Care Management', 
        'Management of Labor and Employment Risks', 
        'Employee Satisfaction Surveys', 
        'Development and Promotion of Employee Ethics Compliance Policies', 
        'Investigation of Ethics Compliance Violations', 
        'Protection of Employee Personal Data', 
        'Development of Violation Grading Standards and Accountability Procedures', 
        'Accountability for Violations', 
        'Maintenance of Employee Information in SHR Systems', 
        'Work Time Management', 
        'Leave Management', 
        'Management of Attendance Irregularities', 
        'Management of Attendance Rules Modifications', 
        'Payroll Ledger Management', 
        'Payroll Account Management', 
        'Management of Labor Contract Templates', 
        'Onboarding Procedures', 
        'Offboarding Procedures', 
        'Employee Record Services', 
        'Employee Inquiries', 
        'Card and Certificate Services', 
        'Proof Services', 
        'Development/Maintenance of Training Policies and Processes', 
        'Training Needs Analysis and Forecasting (including annual training needs)', 
        'Compilation of Training Plans and Budgets (including annual training plans and budgets)', 
        'Training Project Design', 
        'Course Design and Development', 
        'Organization and Implementation of Training (including project and course execution)', 
        'Training Assessment and Evaluation', 
        'Construction and Maintenance of Course Systems (including learning maps)', 
        'Training Venue and Facility Management', 
        'Training Expense Applications', 
        'Signing of Training Agreements', 
        'Development and Collaboration with External Training Resources (including institutions, lecturers, etc.)', 
        'Certification Services (including internal/external professional qualification recognition)', 
        'Establishment and Maintenance of Training Records (including employee and organizational records)', 
        'Internal Trainer Management (selection, training, utilization, retention, and elimination)', 
        'Mentor Management (selection, training, utilization, retention, and elimination)', 
        'Establishment and Maintenance of Training Resource Libraries', 
        'Development and Maintenance of Internal Online Learning Resources', 
        'Operation and Maintenance of Learning Platforms', 
        'Management of External Online Learning Resources', 
        'Online Learning and Operation Activities', 
        'Account Setup and Maintenance', 
        'Account User Permission Management'
],
    'HRCOE Score': [1.44, 0.72, 0.72, 0.72, 1.44, 1.44, 0.72, 0.72, 1.44, 1.44, 0, 0, 1.44, 0, 0, 0, 1.44, 1.44, 0.72, 0, 1.44, 1.44, 0, 0, 0, 1.44, 0, 0, 1.44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.44, 0, 1.44, 1.44, 0, 1.44, 0, 0, 0, 1.44, 0, 0, 0, 0, 1.44, 1.44, 0, 0, 0, 0, 1.44, 1.44, 1.44, 0, 0, 0, 0, 0, 0, 1.44, 1.44, 1.44, 0, 0, 0, 1.44, 0, 0, 0, 1.44, 0, 1.44, 0, 1.44, 0, 0, 0, 1.44, 1.44, 0, 1.44, 1.44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.44, 0, 0, 0.72, 0.72, 0, 0, 1.44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.72, 0, 0.72, 0, 0, 0.72],
    'HRBP Score': [0, 1.44, 1.44, 1.44, 0, 0, 1.44, 1.44, 0, 0.72, 1.44, 1.44, 0.72, 1.44, 1.44, 1.44, 0.72, 0, 0, 1.44, 0.72, 0.72, 1.44, 1.44, 1.44, 0, 1.44, 1.44, 0.72, 1.44, 0.72, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 0.72, 1.44, 0.72, 0.72, 1.44, 0, 1.44, 1.44, 1.44, 0, 1.44, 1.44, 1.44, 1.44, 0.72, 0, 1.44, 1.44, 1.44, 1.44, 0, 0, 0, 0, 0, 1.44, 0, 1.44, 0, 0, 0.72, 0, 1.44, 1.44, 1.44, 0.72, 1.44, 1.44, 1.44, 0, 1.44, 0, 1.44, 0, 1.44, 1.44, 1.44, 0.72, 0, 1.44, 0, 0, 1.44, 0, 0, 0, 0, 0.72, 0, 0, 0, 0.72, 0.72, 0, 0.72, 0, 0, 0, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 0.72, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44],
    'HRSSC Score': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.44, 0, 0.72, 0, 0, 0, 0, 0, 0, 0, 0, 0.72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.72, 1.44, 1.44, 0, 1.44, 0, 1.44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
})

st.image('tdg.png')

# Streamlit app configuration
st.markdown("<h2 style='font-size: 24px;'>TDG人力资源三支柱落位工具</h2>", unsafe_allow_html=True)

# Step 1: User inputs employee responsibilities in Chinese
st.markdown("<h2 style='font-size: 18px;'>请输入您的职责</h2>", unsafe_allow_html=True)
input_responsibilities = st.text_area(
    "每条职责一行", 
    "")

if st.button("开始落位分析"):
    if input_responsibilities.strip():
        # Translate the Chinese responsibilities to English using deep_translator
        translated_responsibilities = [GoogleTranslator(source='zh-CN', target='en').translate(resp.strip()) for resp in input_responsibilities.splitlines() if resp.strip()]
        
        # Step 2: Analyze and map responsibilities
        #st.header("Mapped Responsibilities")
        
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
        #results_df = pd.DataFrame(mapped_results)
        #st.write(results_df)
        
        # Step 3: Calculate percentages for HR pillars
        total_score = sum(pillar_scores.values())
        if total_score > 0:
            for pillar in pillar_scores:
                pillar_scores[pillar] = (pillar_scores[pillar] / total_score) * 100
        
        # Create a donut chart for pillar scores
        st.header("结果")
        fig = px.pie(
            values=list(pillar_scores.values()),
            names=list(pillar_scores.keys()),
            title="百分比",
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
        st.success(f"经计算，您目前的职责更贴近 {dominant_pillar} ")
  
