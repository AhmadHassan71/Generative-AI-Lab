# src/rag_chain/llm_openai.py

import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def encode_image(image_path):
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_openai_llm(model_name="gpt-4o-mini", max_tokens=256):
    """
    Initialize OpenAI client with the specified model.
    Defaults to GPT-4o Mini.
    
    Returns a function that takes a prompt and returns the model's response.
    """
    # Initialize the OpenAI client with API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate_response(prompt, temp=0.0, image_path=None):
        """
        Generate a response using the OpenAI API.
        
        Args:
            prompt (str): The input prompt for the model
            temp (float): Temperature parameter for generation
            image_path (str): Optional path to an image to include with the prompt
            
        Returns:
            str: The generated text
        """
        try:
            messages = []
            
            # If an image is provided, add it to the message content
            if image_path and os.path.exists(image_path):
                # Encode the image to base64
                base64_image = encode_image(image_path)
                
                # Create a message with image content
                messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }]
            else:
                # Text-only message
                messages = [{"role": "user", "content": prompt}]
            
            # Call the API
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")
            return f"Error: {str(e)}"
    
    return generate_response