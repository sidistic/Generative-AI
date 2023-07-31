from transformers import pipeline

generator = pipeline("text-generation", model = "meta-llama/Llama-2-70b-chat-hf")

res = generator(
    "This is a tutorial for text generations. I live in seattle",
    max_length = 200
)

print(res)