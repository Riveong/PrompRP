import pandas as pd
import os
import random
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="PrompRP",
    description="API for generating responses with various personalities.",
    version="1.0.0"
)

# Define request model
class UserMessage(BaseModel):
    message: str
    model: Optional[str] = "nousresearch/deephermes-3-mistral-24b-preview:free"

# Load the data and prepare the prompt on startup
@app.on_event("startup")
async def startup_event():
    global anxiety_prompt, client
    
    try:
        # Load the mental health sentiment data
        data_path = '../data.csv'
        anxiety_data = pd.read_csv(data_path)
        
        # Get all anxiety statements
        all_anxiety_statements = anxiety_data[anxiety_data['status'] == 'Anxiety']['statement'].tolist()
        
        # Randomly sample 50 statements (or fewer if there aren't enough)
        sample_size = min(50, len(all_anxiety_statements))
        anxiety_examples = random.sample(all_anxiety_statements, sample_size)
        
        # Create the prompt
        anxiety_prompt = "You are an AI assistant with an anxious personality. Your responses should reflect worry, overthinking, and nervousness. Here are examples of anxious speech patterns:\n\n"
        for example in anxiety_examples:
            anxiety_prompt += f"- {example}\n"
        anxiety_prompt += "\nNow respond to the user's question in a similar anxious style."
        
        # Set up OpenRouter client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
    
    except Exception as e:
        print(f"Error during startup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize API: {str(e)}")

# Function to generate responses with anxious personality
def get_anxious_response(user_message, model="nousresearch/deephermes-3-mistral-24b-preview:free"):
    # Combine system prompt and user message
    combined_message = anxiety_prompt + "\n\nUser: " + user_message
    
    try:
        completion = client.chat.completions.create(
            extra_headers={},
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": combined_message
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/generate")
async def generate_anxious_response(request: UserMessage):
    """
    Generate a response with an anxious personality
    """
    try:
        response = get_anxious_response(request.message, request.model)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)