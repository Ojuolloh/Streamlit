import streamlit as st
import pandas as pd


def home_page():
    st.title("Classification Project")
    st.write("Telco Churn Classification Project")

    st.markdown("""This uses machine learning to classify whether a customer is likely to churn or not""")
    st.subheader("Key features")
    st.markdown("""
    - Upload your CSV file containing patient data
                
    - Select the desired features for Classification
                
    - Choose a machine learning model from the dropdown menu
                
    - Click classify to get the predicted result
                
    - The App also provides a detailed report on the performance of the model
                
    - The report includes metrics like accuracy,precision,recall,and F1 score
            """)
    st.subheader("App features")
    st.markdown("""
    - **Data View**: Accesses the customer Data
    - **Predict view**:shows various models & predictions you make
    - **Dasboard**: Shows Data Visualizations
""")
    
    st.subheader("User Benefits")
    st.markdown("""        
    - **Data Driven Decisions**: You make an informed decision backed by data          
    - **Access Machine learning**: Utilize Machine learning Algorithm
        """)
    
    st.write('#### How to run the Application')
    with st.container(border=True):
        st.code("""
        # Activate the virtual environment
        env/scripts/activate
                
        # Run the App
        Streamlit run p.py
                """)
        
        # adding the embeded link
        #st.video("https://www.youtube.com/watch?v=wOBBkIQTh1E",autoplay=True)

        # adding the clickable link
        #st.markdown("[watch a demo](https://www.youtube.com/watch?v=wOBBkIQTh1E)")

        # add an image
        st.image(r"https://unsplash.com/photos/I2UR7wEftf4")

        st.divider()
        st.write("====" * 10)
        st.write("Need help? Contact me on +254700400500")



