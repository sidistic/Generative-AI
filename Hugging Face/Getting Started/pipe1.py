from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res =  classifier("I might not be able to get a job")

print(res)