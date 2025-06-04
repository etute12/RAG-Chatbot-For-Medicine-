import streamlit as st
import time
import os
import google.generativeai as genai
from google.generativeai import types

# Set up Google API key
def setup_google_api():
    # Check for API key in environment variables first
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # If not found in environment, try to get from session state or prompt user
    if not api_key:
        if "google_api_key" in st.session_state:
            api_key = st.session_state.google_api_key
        else:
            api_key = st.sidebar.text_input("Enter your Google AI API Key:", type="password")
            if api_key:
                st.session_state.google_api_key = api_key
    
    if not api_key:
        st.sidebar.warning("Please enter your Google AI API key to continue")
        return False
    
    # Initialize Google AI client
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.sidebar.error(f"Error initializing Google AI: {str(e)}")
        return False

# Function to generate response with retry logic
def generate_response_with_retry(prompt, conversation_history=None, max_retries=3, retry_delay=2):
    """
    Generate a response with automatic retries for overloaded model errors.
    """
    retries = 0
    while retries <= max_retries:
        try:
            # Initialize the model with system instructions directly
            generation_config = {
                "temperature": 0.3,  # Reduced for more focused responses
                "top_p": 0.8,       # Reduced for more precise answers
                "top_k": 40,        # Reduced for more focused responses
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Updated system instruction for concise responses
            system_instruction = """You are a healthcare assistant. Be CONCISE and DIRECT.

INTERACTION FLOW:
1. Ask for name
2. Ask for age  
3. Ask for gender
4. Ask for location (city/area)
5. Ask about symptoms/issue
6. Provide brief diagnosis and treatment

RESPONSE FORMAT:
- Keep responses under 3 sentences
- Be direct and professional
- No lengthy explanations
- Focus on essential information only

DIAGNOSIS FORMAT:
- Possible condition: [brief diagnosis]
- First aid: [2-3 key steps]
- Medication: [common treatments]
- See a doctor immediately for proper diagnosis
- Nearby hospitals: [provide 1-2 options based on location]

LOCATION-BASED HOSPITALS:
- Lagos: Lagos University Teaching Hospital, General Hospital Lagos
- Abuja: University of Abuja Teaching Hospital, National Hospital Abuja  
- Covenant University area: Covenant University Hospital https://maps.app.goo.gl/njSWK8Gj8JPmjv5SA
- Port Harcourt: University of Port Harcourt Teaching Hospital
- Kano: Aminu Kano Teaching Hospital
- Ibadan: University College Hospital Ibadan

Ask for location to provide specific nearby hospitals."""
            
            if conversation_history:
                # Create chat with existing history
                chat = model.start_chat(history=conversation_history)
                response = chat.send_message(prompt, stream=True)
            else:
                # First message - create a chat and prepend our instructions
                chat = model.start_chat()
                
                # For the first message, prefix with our instructions
                prefixed_prompt = f"{system_instruction}\n\nUser: {prompt}"
                response = chat.send_message(prefixed_prompt, stream=True)
            
            # Return the streaming response object and the chat object
            return response, chat
            
        except Exception as e:
            error_message = str(e)
            
            # If it's an overloaded model error, retry
            if "503" in error_message and "overloaded" in error_message.lower():
                retries += 1
                if retries <= max_retries:
                    st.warning(f"Model overloaded. Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
                else:
                    st.error("Maximum retry attempts reached. The service is currently unavailable.")
                    return None, None
            else:
                # For other errors, don't retry
                st.error(f"Error generating response: {error_message}")
                return None, None

# Function to generate response (keeping original name for compatibility)
def generate_response(prompt, conversation_history=None):
    return generate_response_with_retry(prompt, conversation_history)

# Main app
def main():
    st.title("Healthcare Assistant")
    st.caption("Quick medical guidance - Always consult a doctor for proper diagnosis")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None
    
    # Configure sidebar
    st.sidebar.title("Settings")
    st.sidebar.info("This chatbot provides preliminary health guidance. Always consult healthcare professionals for accurate diagnosis and treatment.")
    
    # Setup Google API
    if not setup_google_api():
        st.stop()
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Auto-start the conversation if it's empty
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            initial_message = "Hello! I'm your healthcare assistant. What's your name?"
            st.markdown(initial_message)
        st.session_state.messages.append({"role": "assistant", "content": initial_message})
    
    # Handle user input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare the conversation history in the format expected by Google Gemini
        history = []
        for msg in st.session_state.messages[:-1]:  # All except the latest user message
            role = "model" if msg["role"] == "assistant" else "user"
            history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Get streaming response from model
        with st.chat_message("assistant"):
            full_response = ""
            message_placeholder = st.empty()
            
            # Generate response
            if st.session_state.chat_session:
                # Use existing chat session
                response, chat = generate_response(prompt, history)
            else:
                # Create a new chat session
                response, chat = generate_response(prompt)
                if chat:
                    st.session_state.chat_session = chat
            
            # Stream the response
            if response:
                try:
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                    
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    error_message = f"Error while streaming response: {str(e)}"
                    message_placeholder.markdown(error_message)
                    full_response = error_message
            else:
                message_placeholder.markdown("I'm sorry, I couldn't generate a response. Please try again.")
                full_response = "I'm sorry, I couldn't generate a response. Please try again."
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()