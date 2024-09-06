from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import torch
from typing import List, Dict, Optional
import uuid

app = FastAPI()

# Load the tokenizer and model
model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Print model configuration
print("Model Configuration:")
print(model.config)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = Field(default=None, description="Unique identifier for the conversation")

class Conversation:
    def __init__(self):
        self.messages: List[str] = []
        self.chat_history_ids: Optional[torch.Tensor] = None

conversations: Dict[str, Conversation] = {}

def get_or_create_conversation(conversation_id: Optional[str] = None) -> tuple[Conversation, str]:
    if conversation_id is None or conversation_id not in conversations:
        new_id = str(uuid.uuid4())
        conversations[new_id] = Conversation()
        return conversations[new_id], new_id
    return conversations[conversation_id], conversation_id

def generate_response(conversation: Conversation, new_message: str) -> str:
    print(f"Input message: {new_message}")

    # Reset conversation history for each new message
    conversation.chat_history_ids = None

    # Check if it's a translation request
    if "translate" in new_message.lower():
        to_translate = new_message.split("translate")[-1].strip().strip('"').strip("'")
        target_language = "German"  # Default to German, but you could extract this from the message too
        prompt = f"Translate the following to {target_language}. Only provide the translation, nothing else:\n{to_translate}\n\nTranslation:"
    else:
        prompt = f"Human: {new_message}\nAI:"

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)

    print(f"Encoded input: {tokenizer.decode(inputs.input_ids[0])}")

    # Generate a response with adjusted parameters
    output_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=500,  # Reduced max_length for more concise responses
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,      
        do_sample=True,
        top_k=60,
        top_p=0.95,
        temperature=0.6,
        repetition_penalty=1.5
    )
   
    print(f"Raw output from model: {output_ids}")

    # Decode the response
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the input prompt and AI prefix from the response
    response = full_response.split("AI:")[-1].strip()
    
    # For translation requests, extract only the translated part
    if "translate" in new_message.lower():
        response = response.split("Translation:")[-1].strip()
   
    print(f"Final response: {response}")

    conversation.messages.append(new_message)
    conversation.messages.append(response)
   
    return response
     
@app.post("/generate/")
async def generate_response_endpoint(request: MessageRequest):
    try:
        conversation, conversation_id = get_or_create_conversation(request.conversation_id)
        ai_response = generate_response(conversation, request.text)
        return {"response": ai_response, "conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id in conversations:
        return {"messages": conversations[conversation_id].messages}
    raise HTTPException(status_code=404, detail="Conversation not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)