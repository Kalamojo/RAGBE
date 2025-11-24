from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

def main():
    #model = "gemini-2.5-flash"
    model = "gemini-2.5-pro"
    client = genai.Client(api_key=gemini_api_key)

    question = "What's up fr?"
    response = client.models.generate_content(
        model=model, contents=question
    )

    print(response.text)

    # response = client.models.list()
    # print("Available models:")
    # print(response.page)

if __name__ == "__main__":
    main()
