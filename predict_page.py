import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
le_sex = data['le_sex']

#pclass, sex, age, sbsp, parch, fare

def show_predict_page():
    st.set_page_config(page_title="Sink or Swim", page_icon=":ship:")
    st.title('Sink or Swim :ship:')
    st.subheader('Would you have survived the RMS Titanic?')
    st.write("The sinking of the RMS Titanic is the deadliest peacetime maritime disaster to date, and resulted in the loss of over 1500 people. Using machine learning (a decision tree regressor model if you're curious,) we can predict whether *you* would sink or swim.")

    pclass = st.select_slider('What class were you riding in? (1st, 2nd, 3rd)',[1,2,3])
    sex = st.selectbox('Sex:', ['male', 'female'])
    age = st.number_input('Age:', 0,100)
    sbsp = st.slider('How many siblings and/or spouses are aboard with you?', 0,10)
    parch = st.slider('How many parents and/or children are aboard with you?', 0,2)
    fare = st.number_input('What did your ticket cost? (The average price across all classes combined was 34 dollars.)',0,500)

    def predict(): 
        X = np.array([[pclass, sex, age, sbsp, parch, fare]])
        X[:, 1] = le_sex.transform(X[:,1])
        X = X.astype('float')

        survived = regressor.predict(X)
        if survived == 1:
            st.success('You survived! :thumbsup:')
        else:
            st.error('Perished... :shark:')

    st.button('Predict', on_click=predict)

    left, center, right = st.columns(3)

    with left:
        st.write("[Check out my GitHub >](https://github.com/justintricate)")
    with center:
        st.write("[This project's repo >](https://github.com/justintricate/sink-or-swim)")
    with right:
        st.write("[Learn about the Titanic >](https://en.wikipedia.org/wiki/Titanic)")

        
