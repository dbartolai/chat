from openai import OpenAI

from dotenv import load_dotenv

# Load variables from the .env file into os.environ
load_dotenv() 


client = OpenAI()

response = client.responses.create(
    model = "gpt-5-nano",
    input = "what's your favorite book? explain why in one sentence."
)

print(response.output_text)