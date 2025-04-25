import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="https://img.icons8.com/emoji/48/bar-chart-emoji.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme
alt.themes.enable("dark")

# Load data
import os

# Check if files exist before loading
cleaned_file_path = r"D:\DEPI Graduation Project\New folder\Final_train_df_cleaned.csv"
test_file_path = r"D:\DEPI Graduation Project\New folder\kaggle_test_prediction_all.csv"

if os.path.exists(cleaned_file_path):
    df_cleaned = pd.read_csv(cleaned_file_path)
else:
    st.error(f"Error: File not found at {cleaned_file_path}")

if os.path.exists(test_file_path):
    df_test = pd.read_csv(test_file_path)
else:
    st.error(f"Error: File not found at {test_file_path}")
@st.cache_data
def load_data():
    df = pd.read_csv("D:\DEPI Graduation Project\dina\Final train_df_cleaned.csv")
    df['song_completion_rate'] = df['num_100'] / (
        df['num_25'] + df['num_50'] + df['num_75'] + df['num_985'] + df['num_100'])
    df['song_completion_rate'] = df['song_completion_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    df['avg_secs_per_song'] = df['total_secs'] / df['num_unq']
    df['avg_secs_per_song'] = df['avg_secs_per_song'].replace([np.inf, -np.inf], 0).fillna(0)
    df['high_engagement'] = (df['total_secs'] > df['total_secs'].median()).astype(int)
    df['membership_days'] = (
        pd.to_datetime(df['membership_expire_date']) - pd.to_datetime(df['transaction_date'])).dt.days.fillna(0)
    df['discount_rate'] = 1 - (df['actual_amount_paid'] / df['plan_list_price'])
    df['discount_rate'] = df['discount_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    df['is_trial'] = (df['actual_amount_paid'] == 0).astype(int)
    df['auto_renew_but_cancel'] = ((df['is_auto_renew'] == 1) & (df['is_cancel'] == 1)).astype(int)
    df['age_group'] = pd.cut(df['bd'], bins=[10, 20, 30, 40, 50, 60, 100], labels=False, right=False)
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m%d', errors='coerce')
    df['account_age_days'] = (
        pd.to_datetime("2017-03-01") - df['registration_init_time']).dt.days.fillna(0)
    df['has_transaction'] = df['payment_method_id'].notna().astype(int)
    df['no_txn_and_churn'] = ((df['has_transaction'] == 0) & (df['is_churn'] == 1)).astype(int)
    df['payment_method_risk'] = df['payment_method_id'].map({
        18: 'low_risk', 11: 'low_risk', 31: 'low_risk',
        3: 'high_risk', 6: 'high_risk', 13: 'high_risk', 22: 'high_risk', -1: 'no_txn'})
    df['payment_method_risk'] = df['payment_method_risk'].fillna('medium_risk')
    df['gender'] = df['gender'].fillna('other')
    df['registered_via'] = df['registered_via'].fillna(-1).astype(int)
    df['payment_method_id'] = df['payment_method_id'].fillna(-1).astype(int)
    df['city'] = df['city'].fillna(-1).astype(int)
    df['age_bucket'] = pd.cut(df['bd'], bins=[13, 18, 25, 35, 45, 55, 65, 75, 85, 95])
    return df

df = load_data()
# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Data Exploration","Feature Engineering ","Model Comparison", "Predict Churn"])

if selection == "Home":
    st.title("📊 Customer Churn Prediction Dashboard")
    st.subheader("Introduction:")
    st.markdown("""
    This dashboard aims to analyze customer data and apply machine learning models to predict the likelihood of customer churn, assisting businesses in making proactive decisions to improve customer retention.
    """)

    # Divide the page into two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Project Objective")
        st.markdown("""
        - Analyze customer data to understand factors influencing churn.
        - Build machine learning models to predict the likelihood of customer churn.
        - Provide an interactive interface to display results and offer recommendations.
        """)

    with col2:
        st.subheader("📁 Dataset Description")

        st.markdown("""
        - **Source**: [Kaggle - Customer Churn Prediction Dataset](https://www.kaggle.com/competitions/customer-retention-datathon-riyadh-edition/data)

        - **Key Fields**:
           - `msno`: Unique customer identifier.
           - `is_churn`: Customer churn status — **1** indicates the customer churned, **0** means they renewed.
           - `bd`: Customer age — note that this column contains **outlier values** ranging from **-7000 to 2015**, which should be cleaned.
           - `payment_method_id`: Encoded payment method used by the customer.
           - `num_unq`: Number of **unique songs** played by the customer.
        """)




    # Summary of models used
    st.subheader("🤖 Summary of Models Used")
    st.markdown("""
    Four machine learning models were employed to predict customer churn, each selected for its specific strengths:

    1. **Random Forest**: An ensemble learning method that constructs multiple decision trees and combines their predictions. It reduces overfitting and improves accuracy by leveraging the "wisdom of the crowd".

    2. **Logistic Regression**: A simple and interpretable statistical model used for binary classification. It estimates the probability of churn based on the relationship between input features and the target variable.

    3. **Decision Tree**: A tree-based model that splits data into branches based on feature values. It's easy to understand and can handle non-linear relationships effectively.

    4. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, assuming independence between features. It's known for its efficiency and effectiveness, especially with high-dimensional data.

    Each model was trained and evaluated using the available dataset. Their predictions are included in the dashboard for performance comparison and insight generation.
    """)
   
    
elif selection == "Data Exploration":
    st.title("🔍 Data Exploration")
    st.markdown("Explore the cleaned dataset and gain insights through interactive visualizations.")
    
    # Show raw data
    st.subheader("📄 Preview of Cleaned Dataset")
    st.dataframe(df_cleaned.head(20))
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.write(df_cleaned.describe())

    # Churn Distribution
    df_cleaned['churn_label'] = df_cleaned['is_churn'].map({0: 'Not Churned', 1: 'Churned'})

    # Plot pie chart
    st.subheader("🔁 Churn Distribution")
    churn_counts = df_cleaned['churn_label'].value_counts()
    fig1 = px.pie(names=churn_counts.index, values=churn_counts.values, 
              title='Churn vs Non-Churn Customers', 
              color_discrete_sequence=['#00cc96', '#ff4b4b'])

    fig1.update_traces(textinfo='percent+label', pull=[0.1] * len(churn_counts))
    st.plotly_chart(fig1)

    # Churn Rate by Payment Method
    df_cleaned['payment_method_id'] = df_cleaned['payment_method_id'].fillna(-1).astype(int)
    payment_method_churn = df_cleaned.groupby('payment_method_id')['is_churn'].mean().reset_index()
    payment_method_churn = payment_method_churn.sort_values(by='payment_method_id', ascending=True)

    # Plot churn rate for each payment method
    st.subheader("💳 Churn Rate by Payment Method")
    fig2, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='payment_method_id', y='is_churn', data=payment_method_churn, palette='viridis', ax=ax)
    ax.set_xlabel('Payment Method ID')
    ax.set_ylabel('Churn Rate')
    ax.set_title('Churn Rate by Payment Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
    
    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    numeric_features = df_cleaned.select_dtypes(include=np.number)
    corr = numeric_features.corr()
    fig3, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig3)
    
    # Histograms for numeric features
    st.subheader("📈 Feature Distributions")
    selected_column = st.selectbox("Select a numeric column to visualize:", numeric_features.columns)
    fig4 = px.histogram(df_cleaned, x=selected_column, color='is_churn', barmode='overlay',
                        title=f'Distribution of {selected_column} by Churn Status',
                        color_discrete_map={0: "#00cc96", 1: "#ff4b4b"})
    st.plotly_chart(fig4)

    
elif selection == "Feature Engineering ":
    st.title("🧪 Feature Engineering Insights")
    

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Song Completion Rate vs. Churn")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='is_churn', y='song_completion_rate', data=df, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Avg Seconds per Song vs. Churn")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='is_churn', y='avg_secs_per_song', data=df, ax=ax2)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Discount Rate by Churn")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='is_churn', y='discount_rate', data=df, ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.subheader("Membership Duration vs. Churn")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='is_churn', y='membership_days', data=df, ax=ax4)
        st.pyplot(fig4)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Churn Rate by Age Group")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='age_group', y='is_churn', data=df, estimator='mean', ax=ax5)
        st.pyplot(fig5)

    with col6:
        st.subheader("Payment Method Risk vs. Churn")
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='payment_method_risk', y='is_churn', data=df, estimator='mean', ax=ax6)
        st.pyplot(fig6)


   




elif selection == "Model Comparison":
    st.title("Model Comparison")
    st.subheader("🤖 Model Performance Comparison")
    
    pred_file_path = r"D:/DEPI Graduation Project/New folder/all_predictions.csv"
    df_pred = pd.read_csv(pred_file_path)
    
    # Display the metrics for each model in the dataset
    st.subheader("📊 Raw Model Predictions")
    st.dataframe(df_pred.head(10))

   
    # Load model comparison results
    comparison_file_path = r"D:\DEPI Graduation Project\New folder\model_comparison_results.csv"
    df_models = pd.read_csv(comparison_file_path)
    
    # Display model comparison table
    st.subheader("📋 Evaluation Metrics for All Models")
    st.dataframe(df_models)
    
    # باقي كود البار شارت والدونات...

    # Choose the model for detailed evaluation
    st.subheader("🔎 Select a Model to Analyze")
    # Define model_columns mapping (move this to a global scope for reuse)
    model_columns = {
        "Random Forest": "Predicted_Probability_RF",
        "Logistic Regression": "Predicted_Probability_LR",
        "Decision Tree": "Predicted_Probability_DT",
        "Naive Bayes": "Predicted_Probability_GNB"
    }

# اختيار الموديل
    selected_model = st.selectbox("Select Model", list(model_columns.keys()))
 
# استخراج القيم الحقيقية وقيم الاحتمال
    y_true = df_pred["True_Label"]
# Ensure model_columns is accessible here
    if selected_model in model_columns and model_columns[selected_model] in df_pred.columns:
      y_scores = df_pred[model_columns[selected_model]]
    else:
      st.error(f"خطأ: العمود '{model_columns[selected_model]}' غير موجود في البيانات.")
      st.stop()

# حساب منحنى ROC و AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

# رسم ROC باستخدام Plotly
    fig = go.Figure()

# منحنى ROC
    fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name='ROC Curve',
    line=dict(color='royalblue', width=3)
    ))

# الخط العشوائي
    fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Guess',
    line=dict(color='red', width=2, dash='dash')
    ))

# عرض AUC داخل الرسم
    fig.add_annotation(
    x=0.6,
    y=0.1,
    text=f"AUC = {roc_auc:.2f}",
    showarrow=False,
    font=dict(size=16, color="black"),
    bgcolor="lightyellow",
    bordercolor="black",
    borderwidth=1
)

# تخصيص شكل الرسم
    fig.update_layout(
    title=dict(text=f"<b>ROC Curve - {selected_model}</b>", x=0.5),
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    width=550,
    height=600,
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    template='plotly_white',
    legend=dict(x=0.7, y=0.3)
    )

# عرض الرسم
    st.plotly_chart(fig, use_container_width=False)


    # Load model comparison results
    comparison_file_path = r"D:\DEPI Graduation Project\New folder/model_comparison_results.csv"
    df_models = pd.read_csv(comparison_file_path)

    # Display model comparison table
    st.subheader("📋 Evaluation Metrics for All Models")
    st.dataframe(df_models)

    # Bar chart for comparison
    st.subheader("📊 Comparison of Model Performance")


# Create bar chart with Plotly
    fig = go.Figure()

# Add bars for each metric with customized colors
    fig.add_trace(go.Bar(
    x=df_models['Model'], 
    y=df_models['Accuracy'], 
    name='Accuracy', 
    marker_color='#1f77b4'  # Blue for Accuracy
    ))
    fig.add_trace(go.Bar(
    x=df_models['Model'], 
    y=df_models['Precision'], 
    name='Precision', 
    marker_color='#2ca02c'  # Green for Precision
    ))
    fig.add_trace(go.Bar(
    x=df_models['Model'], 
    y=df_models['Recall'], 
    name='Recall', 
    marker_color='#ff7f0e'  # Orange for Recall
    ))
    fig.add_trace(go.Bar(
    x=df_models['Model'], 
    y=df_models['F1-Score'], 
    name='F1-Score', 
    marker_color='#d62728'  # Red for F1-Score
    ))
    fig.add_trace(go.Bar(
    x=df_models['Model'], 
    y=df_models['AUC'], 
    name='AUC', 
    marker_color='#9467bd'  # Purple for AUC
    ))

# Customize layout for better appearance
    fig.update_layout(
    barmode='group',  # Group bars together
    title="Model Performance Comparison",
    xaxis_title="Model",
    yaxis_title="Scores",
    legend_title="Metrics",
    template="plotly_white",  # Clean white background
    width=600,  # Width of the chart
    height=400  # Height of the chart (adjusted here)
    )

# Show the chart in Streamlit
    st.plotly_chart(fig)

    # plot donut chart for model comparison
    st.subheader("🍩 Model Performance Donut Chart")

# ألوان مخصصة لكل موديل
    colors = colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


    fig = go.Figure(data=[
    go.Pie(
        labels=df_models['Model'],
        values=df_models['AUC'],
        hole=0.4,
        textinfo='label+percent+value',
        marker=dict(colors=colors, line=dict(color='#000000', width=2)),
        hoverinfo='label+value+percent'
    )
    ])

    fig.update_layout(
    title_text="Model Performance (AUC)",
    title_font_size=20,
    template="plotly_white",
    showlegend=True,
    annotations=[dict(text='AUC', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig)

  # Confusion Matrix for the selected model
    st.subheader("📉 Confusion Matrix")

# Model to prediction column mapping
    prediction_columns = {
    "Random Forest": "Predicted_Label_RF",
    "Logistic Regression": "Predicted_Label_LR",
    "Decision Tree": "Predicted_Label_DT",
    "Naive Bayes": "Predicted_Label_GNB"
}

# Select model
    selected_model = st.selectbox("Select Model for Confusion Matrix", list(prediction_columns.keys()))

# Get the true and predicted labels
    y_true = df_pred["True_Label"]

    if selected_model in prediction_columns:
     y_pred = df_pred[prediction_columns[selected_model]]
    else:
     st.error(f"Error: Prediction column for model '{selected_model}' not found.")
     st.stop()

# Compute confusion matrix


# Create heatmap
# حساب Confusion Matrix والنسب
# Create confusion matrix and its percentage display
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_display = np.array([[f"{count}<br>({percent:.0%})" for count, percent in zip(row1, row2)]
                       for row1, row2 in zip(cm, cm_percent)])

# Create Heatmap
    fig_cm = ff.create_annotated_heatmap(
    z=cm,
    annotation_text=cm_display,
    x=['Predicted Not Churned', 'Predicted Churned'],
    y=['Actual Not Churned', 'Actual Churned'],
    colorscale='YlGnBu',  # You can change this to something like 'Viridis' or 'Cividis'
    showscale=True,
    font_colors=["black"]  # Set font color for better contrast (alternating between white and black)
    )

    # Improve layout
    fig_cm.data[0].colorbar.title = 'Count'

    fig_cm.update_layout(
    title=dict(
        text=f"<b>Confusion Matrix - {selected_model}</b>",
        x=0.5,
        font=dict(size=20),
        y=0.99  # Lift the title a bit
    ),
    xaxis=dict(
        title="Predicted Label",
        showgrid=False,
        tickfont=dict(size=14),
        title_standoff=10  # Adjust distance between title and axis
    ),
    yaxis=dict(
        title="Actual Label",
        showgrid=False,
        tickfont=dict(size=14),
        title_standoff=10
    ),
    template="plotly_white",  # Change background template
    width=600,
    height=500,
    margin=dict(l=60, r=60, t=100, b=60)  # Bigger margins, especially on top
    )

# Display the confusion matrix plot
    st.plotly_chart(fig_cm)


elif selection == "Predict Churn":
   st.title("🔮 Predict Customer Churn")
   st.markdown("Use the trained model to predict whether a customer is likely to churn.")

# Load the model
   model_path = r"D:\DEPI Graduation Project\New folder\random_forest_model.pkl"
   with open(model_path, 'rb') as file:
    rf_model = pickle.load(file)

# File uploader UI in English
   uploaded_file = st.file_uploader("📁 Upload customer data file (CSV)", type=["csv"])
   if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

