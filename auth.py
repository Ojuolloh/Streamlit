import streamlit as st

# Define credentials (replace with secure storage for production)
users = {
    "admin": "password123",
    "user1": "password456",
}

def login():
    """Displays login form and verifies credentials."""
    st.title("Login Page")
    
    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    # Login button
    if st.button("Login"):
        # Verify credentials
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.success(f"Welcome, {username}!")
            
            # Use experimental_set_query_params to trigger a reload
            st.experimental_set_query_params(logged_in="true")
        else:
            st.error("Invalid username or password. Please try again.")

def logout():
    """Logs the user out and resets authentication state."""
    st.session_state["authenticated"] = False
    
    # Use experimental_set_query_params to trigger a reload
    st.experimental_set_query_params(logged_in="false")

def authentication_required():
    """Checks if the user is authenticated. If not, displays the login form."""
    if st.session_state.get("authenticated", False):
        return True  # User is authenticated
    else:
        login()  # Show login form if not authenticated
        return False
