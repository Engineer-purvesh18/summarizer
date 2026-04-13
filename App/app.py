from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates # for User Interface part 
from fastapi.responses import HTMLResponse # returning entire html file 
from fastapi.staticfiles import StaticFiles 

# Initializing the fastapi app

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Loading model and tokenizer 
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device for running PyTorch computations

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_availanle():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Templating 

templates = Jinja2Templates(directory=".") # index.html is present in same directory

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

# copied from the main.ipynb file 
def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse) # getting info from APIs
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request}) # returning html templates