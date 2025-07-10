import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load CSV files
@st.cache_data
def load_data():
    users_df = pd.read_csv("real_users.csv")
    courses_df = pd.read_csv("real_courses.csv")
    interactions_df = pd.read_csv("real_interactions.csv")
    return users_df, courses_df, interactions_df

users_df, courses_df, interactions_df = load_data()

# Content-Based Filtering
courses_df['tags'] = courses_df['tags'].fillna('')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(courses_df['tags'])
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
course_indices = pd.Series(courses_df.index, index=courses_df['title']).drop_duplicates()

def get_similar_courses(course_title, top_n=5):
    if course_title not in course_indices:
        return []
    idx = course_indices[course_title]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    course_indices_result = [i[0] for i in sim_scores]
    return courses_df['title'].iloc[course_indices_result].tolist()

# Collaborative Filtering
@st.cache_resource
def train_cf_model():
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(interactions_df[['user_id', 'course_id', 'quiz_score']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model

cf_model = train_cf_model()

def recommend_courses_for_user(user_id, model, top_n=5):
    taken_courses = interactions_df[interactions_df['user_id'] == user_id]['course_id'].tolist()
    all_courses = courses_df['course_id'].tolist()
    unseen_courses = [cid for cid in all_courses if cid not in taken_courses]
    predictions = [(cid, model.predict(user_id, cid).est) for cid in unseen_courses]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_courses = [courses_df[courses_df['course_id'] == cid]['title'].values[0] for cid, _ in predictions[:top_n]]
    return top_courses

# Streamlit UI
st.title("📚 Course Recommendation System")

option = st.radio("Choose Recommendation Type:", ("Content-Based", "Collaborative Filtering"))

if option == "Content-Based":
    selected_course = st.selectbox("Select a course you like:", courses_df['title'].unique())
    if selected_course:
        recommendations = get_similar_courses(selected_course)
        st.subheader("Recommended Courses:")
        for rec in recommendations:
            st.write("-", rec)

elif option == "Collaborative Filtering":
    selected_user_id = st.selectbox("Select your User ID:", users_df['user_id'].unique())
    if selected_user_id:
        recommendations = recommend_courses_for_user(selected_user_id, cf_model)
        st.subheader("Recommended Courses for You:")
        for rec in recommendations:
            st.write("-", rec)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
