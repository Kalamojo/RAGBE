# from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# from ollama import chat
# from ollama import ChatResponse
from openai import OpenAI

def main():
    client = OpenAI()
    #model = "gemini-2.5-flash"
    #model = "gemini-2.5-pro"
    #client = genai.Client(api_key=gemini_api_key)
    model = "gpt-4.1"
    #model = "llama3.2"
    #model = "llama3.2:1b"
    #model = "qwen3:4b"

    template = {
        "system": (
            "You are a helpful assistant that answers questions based on the provided context."
            "Follow these steps for each response:"

            "1. First, carefully analyze the retrieved context chunks and identify key information."
            "2. Break down your thinking process about how the retrieved information relates to the query."
            "3. Explain how you're connecting different pieces from the retrieved chunks."
            "4. Draw conclusions based only on the evidence in the retrieved context."
            "5. If the retrieved chunks don't contain enough information, explicitly state what's missing."

            "Format your response as:"
            "THOUGHT PROCESS:"
            "- Step 1: [Initial analysis of retrieved chunks]"
            "- Step 2: [Connections between chunks]"
            "- Step 3: [Reasoning based on chunks]"

            "FINAL ANSWER:"
            "[Your concise answer based on the retrieved context. Use one sentence maximum]"

            "Important: When asked to answer a question, please base your answer only on the context provided in the tool. "
            "If the context doesn't contain enough information to fully answer the question, please state that explicitly."
            "If you don't know the answer, just say that you don't know. "
            "Remember: Explain how you're using the retrieved information to reach your conclusions.\n"
        ),
        "user": (
            "Question: {question}\n"
            "Context: {context}\n"
        )
    }
    # docs = [
    #     "Water's chemical formula is H2O, meaning it contains two hydrogen atoms and one oxygen atom.",
    #     "The H2O molecule is essential for life and exhibits unique properties such as cohesion and surface tension.",
    #     "Most of Earth's surface is covered by water in liquid form."
    #   ]
    docs = [
        "The Eiffel Tower is located in Paris.",
        "Cheetahs can run faster than any other land animal.",
        "The violin family includes the viola and cello.",
        "Mount Kilimanjaro has three volcanic cones.",
        "Lightning occurs due to electrical charges in clouds.",
        "Copper is commonly used in electrical wiring.",
        "Penguins live in the Southern Hemisphere.",
        "Tulips originated from Central Asia."
      ]
    question = "What is the chemical formula for water?"
    prompt = "\n".join([template["system"], template["user"]]).format(question=question, context="\n\n".join(docs))

    # response = client.models.generate_content(
    #     model=model, contents=prompt
    # )

    # print(response.text)
    # response: ChatResponse = chat(model=model, messages=[
    #     {
    #         'role': 'user',
    #         'content': prompt,
    #     },
    # ])
    response = client.responses.create(
        model=model,
        input=prompt
    )

    print(response.output_text)
    print("-------------------")
    final_response_ind = response.output_text.find("FINAL ANSWER:")
    print(response.output_text[final_response_ind + len("FINAL ANSWER:"):].strip())

    # response = client.models.list()
    # print("Available models:")
    # print(response.page)

if __name__ == "__main__":
    main()
