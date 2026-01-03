from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

# Try common Gemini model names
models_to_try = [
    "gemini-1.5-flash",
    "gemini-1.5-pro", 
    "gemini-pro",
    "gemini-1.0-pro"
]

for model_name in models_to_try:
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = llm.invoke("Hello")
        print(f"✅ {model_name} - WORKS")
        break
    except Exception as e:
        print(f"❌ {model_name} - {str(e)[:100]}...")
