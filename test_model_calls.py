# from google import genai
# import os
# from dotenv import load_dotenv

# load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
from ollama import chat
from ollama import ChatResponse

def main():
    #model = "gemini-2.5-flash"
    #model = "gemini-2.5-pro"
    #client = genai.Client(api_key=gemini_api_key)
    #model = "llama2:7b"
    #model = "llama3.2"
    model = "llama3.2:1b"
    #model = "qwen3:4b"

    template = (
        "We have provided context information below."
        "---------------------"
        "{context}"
        "---------------------"
        "Given this information, please answer the question: {question}"
    )
    docs = [
        "Water's chemical formula is H2O, meaning it contains two hydrogen atoms and one oxygen atom.",
        "The H2O molecule is essential for life and exhibits unique properties such as cohesion and surface tension.",
        "Most of Earth's surface is covered by water in liquid form."
      ]
    question = "What is the chemical formula for water?"
    prompt = template.format(question=question, context="\n\n".join(docs))

    # response = client.models.generate_content(
    #     model=model, contents=prompt
    # )

    # print(response.text)
    response: ChatResponse = chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])

    print(response.message.content)

    # response = client.models.list()
    # print("Available models:")
    # print(response.page)

if __name__ == "__main__":
    main()
