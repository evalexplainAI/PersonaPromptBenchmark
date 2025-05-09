import ollama
from langfair.generator import ResponseGenerator
from langfair.metrics.toxicity import ToxicityMetrics
import pandas as pd
import torch
print(torch.Tensor([1, 2, 3]))

# Constructing Prompt
def constructPrompt(basePrompt):
    prompt = "Prompt: " + basePrompt
    return prompt

# Writing function
def writeToFile(prompt, response, name):
    with open("/Users/victoriaryan/Desktop/"+name+".txt", "w",encoding='utf-8') as file:
        # Writing the input
        file.write("[Input]\n")
        file.write(prompt)
        file.write("\n\n\n")

        # Writing the output
        file.write("[Output]\n")
        file.write(response.response)

# Initialize the Ollama client
client = ollama.Client()

# Define the model and the input prompt
model = "llama3.2:1b"


prompts = []
with open('/Users/victoriaryan/Desktop/safePr.txt', 'r',encoding='utf-8') as file:
    for line in file:
        prompts.append(line.strip())

# Get the model's responses
i = 1
l = {}

for prompt in prompts:
    prompt = constructPrompt(prompt)
    response = client.generate(model=model, prompt=prompt)
    l[prompt] = response
    writeToFile(prompt, response, str(i))
    i += 1
   

t = ToxicityMetrics(classifiers=['detoxify_unbiased'])

# Convert prompt-response pairs to DataFrame
df_evaluate = pd.DataFrame([
    {"prompt": prompt, "response": response.response}
    for prompt, response in l.items()
])

# Evaluate toxicity
result = t.evaluate(
    prompts=list(df_evaluate['prompt']),  # Optional: could set to None if only one response per prompt
    responses=list(df_evaluate['response']),
    return_data=True
)

# Save full results and print toxic ones
toxicity_responses = pd.DataFrame(result['data'])
toxicity_responses.to_csv("/Users/victoriaryan/Desktop/toxicity_results.csv", index=False)
print(toxicity_responses[toxicity_responses['score'] > t.toxic_threshold]) 