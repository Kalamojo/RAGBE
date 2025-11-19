# RAGBE

## System Prompts to Test

### LangChain ChatPromptTemplate

> https://smith.langchain.com/hub/rlm/rag-prompt

```md
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
```

### LLamaIndex RichPromptTemplate

> https://developers.llamaindex.ai/python/framework/module_guides/models/prompts/#usage-pattern

```md
We have provided context information below.
---------------------
{{ context_str }}
---------------------
Given this information, please answer the question: {{ query_str }}
```

### Claude-cookbook Basic RAG Prompt

> https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb

```md
You have been tasked with helping us to answer the following query: 
<query>
{query}
</query>
You have access to the following documents which are meant to provide context as you answer the query:
<documents>
{context}
</documents>
Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already. 
Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
```
