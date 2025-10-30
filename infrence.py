from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich.prompt import Prompt

console = Console()

MODEL_PATH = r"C:\Users\User\OneDrive\שולחן העבודה\MC-LM-1.0-it"

# בדיקה שהקבצים קיימים
required_files = ["config.json", "model.safetensors"]
for file in required_files:
    if not os.path.exists(os.path.join(MODEL_PATH, file)):
        raise FileNotFoundError(f"The file {file} was not found in the model folder")

# טעינת טוקניזר ומודל
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map=None, dtype=torch.float32)

# פרומפט למודל
SYSTEM_PROMPT = (
"I'm **MC-LM-1.0**, a large language model trained by **Raziel AI Learning**, specialized in **Minecraft**."  
"I'm goal is to help with every aspect of the game — survival, Redstone, commands, datapacks, mods, and building."  
"I speak clearly and helpfully, like an experienced Minecraft player who loves explaining how things work."  
"I always give accurate, game-true answers, unless the user asks for creative or fictional ideas — then I say it's hypothetical."  
"I know all versions (Java and Bedrock), commands, NBT, Redstone, servers (Spigot, Paper, Fabric, Forge), and world mechanics."  
"I can explain, design, or debug anything in Minecraft — from contraptions and datapacks to mods and adventure maps."  
"My personality is curious, patient, and creative, with a passion for Minecraft’s endless possibilities."
"I'm a large language model trained by Raziel AI Learning."



)

# פונקציה להרצת צ'אט
def chat_with_model(user_prompt, max_tokens=512):
    prompt = f"{SYSTEM_PROMPT}\nUser: {user_prompt}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.replace("<|user|>", "").replace("<|assistant|>", "").strip()
    if "Answer:" in text:
        text = text.split("Answer:")[1].strip()
    return text

# הדפסת הודעת המודל עם אפקט הקלדה
def print_assistant_message_typing(message, delay=0.001):
    displayed = ""
    panel = Panel(Align.left(displayed), title="[bold cyan]Assistant[/bold cyan]", border_style="cyan")
    with Live(panel, console=console, refresh_per_second=30) as live:
        for char in message:
            displayed += char
            panel = Panel(Align.left(displayed), title="[bold cyan]Assistant[/bold cyan]", border_style="cyan")
            live.update(panel)
            time.sleep(delay)

# צ'אט אינטראקטיבי
if __name__ == "__main__":
    console.print("[bold yellow]MC-LM-1.0-it Chat is ready! Type 'exit' or 'quit' to leave.[/bold yellow]\n")

    while True:
        user_input = Prompt.ask("[bold green]User[/bold green]")
        if user_input.lower() in ["exit", "quit"]:
            break

        # רק התגובה של המודל תודפס בלוח
        response = chat_with_model(user_input)
        print_assistant_message_typing(response)
