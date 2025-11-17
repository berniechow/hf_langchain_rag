from langchain_huggingface import HuggingFacePipeline

#model_id = "tiiuae/falcon-7b-instruct"
model_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "top_k": 50,
        "temperature": 0.1,
    },
)

response = llm.invoke("What is the capital of South Korea?")
print(response)