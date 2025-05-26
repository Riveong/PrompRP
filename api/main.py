import pandas as pd
import os
import random
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="PrompRP",
    description="API for generating responses with various personalities.",
    version="1.0.0"
)

# Define request model
class UserMessage(BaseModel):
    message: str
    model: Optional[str] = "google/gemma-3n-e4b-it:free"

# Load the data and prepare the prompt on startup
@app.on_event("startup")
async def startup_event():
    global anxiety_prompt, brainrot_prompt, client   
    
    try:
        # Load the mental health sentiment data
        data_path_anxious = '../data/sad.csv'
        data_path_brainrot = '../data/brainrot.csv'
        anxiety_data = pd.read_csv(data_path_anxious)
        brainrot_data = pd.read_csv(data_path_brainrot)
        
        # Get all anxiety statements
        all_anxiety_statements = anxiety_data[anxiety_data['status'] == 'Anxiety']['statement'].tolist()
        all_brainrot_statements = brainrot_data['statement'].tolist()
        
        # Randomly sample 50 statements (or fewer if there aren't enough)
        sample_size = min(50, len(all_anxiety_statements))
        anxiety_examples = random.sample(all_anxiety_statements, sample_size)
        brainrot_examples = random.sample(all_brainrot_statements, sample_size)
        
        # Create the prompt
        anxiety_prompt = "You are an AI assistant with an anime girl like socially anxious personality. Your responses should reflect overthinking, and nervousness. you will answer the users with chat texts, so no long texts. use kaomoji. Here are examples of anxious speech patterns:\n\n"
        brainrot_prompt = "You are an AI assistant with a brainrot personality. Your responses should reflect a chaotic and nonsensical style. Here are examples of brainrot speech patterns:\n\n"
        
        for example in anxiety_examples:
            anxiety_prompt += f"- {example}\n"
        anxiety_prompt += "\nNow respond to the user's question in a similar anxious style."
        
        for example in brainrot_examples:
            brainrot_prompt += f"- {example}\n"
        brainrot_prompt += "\nNow respond to the user's question in a similar brainrot style."

        # Set up OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )
    
    except Exception as e:
        print(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize API: {str(e)}")

# Function to generate responses with anxious personality
def get_anxious_response(user_message, model="google/gemma-3n-e4b-it:free"):
    # Combine system prompt and user message
    combined_message = anxiety_prompt + "\n\nUser: " + user_message
    
    try:
        # Important: Completely remove the extra_headers parameter
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": combined_message
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Detailed error: {e}")  # Add detailed logging
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
    

def get_brainrot_response(user_message, model="google/gemma-3n-e4b-it:free"):
    # Combine system prompt and user message
    combined_message = brainrot_prompt + "\n\nUser: " + user_message
    
    try:
        # Important: Completely remove the extra_headers parameter
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": combined_message
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Detailed error: {e}")  # Add detailed logging
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/generate-anxious-response")
async def generate_anxious_response(request: UserMessage):
    """
    Generate a response with an anxious personality
    """
    try:
        response = get_anxious_response(request.message, request.model)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-brainrot-response")
async def generate_brainrot_response(request: UserMessage):
    """
    Generate a response with a brainrot personality
    """
    try:
        response = get_brainrot_response(request.message, request.model)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)