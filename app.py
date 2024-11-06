import streamlit as st
from home import home_page
from data import data_page
from predict import predict_page
from dashboard import dashboard_page
from auth import login, authentication_required

# Initialize session state for authentication at the top of the script
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Assigning to the appropriate pages
def main():
    # Check if the user is authenticated; if not, display login
    if authentication_required():
        # User is authenticated, show sidebar navigation
        st.sidebar.title("Navigator")
        st.sidebar.write("Use this to select between pages")
        
        # Sidebar selection
        page = st.sidebar.selectbox("Navigate", ["Home", "Data", "Predict", "Dashboard"])

        # Route to the selected page
        if page == "Home":
            home_page()
        elif page == "Data":
            data_page()
        elif page == "Predict":
            predict_page()
        elif page == "Dashboard":
            dashboard_page()

if __name__ == "__main__":
    main()
