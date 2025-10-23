from groq import Groq
from json import load, dump
import datetime
import random
import re
from dotenv import dotenv_values
import os
import requests



env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
Assistantname = env_vars.get('Assistantname')
Groqapi = env_vars.get('Groqapi')
User_Github_Link = env_vars.get('My_Github_Link')

client = Groq(api_key=Groqapi)

CHAT_LOG_PATH = os.path.join(
    os.path.expanduser("~"),
    "NISHA", "Nisha_rework", "NISHA_Rework", "data", "ChatLog.json"
)


messages =[]

colors = [
    "\033[93m",  #Yellow
    "\033[92m",  #Green
    "\033[96m",  #Cyan
    "\033[94m",  #Blue
    "\033[95m",  #Magenta
]

System = f"""
Hello, I am {Username}. You are an advanced AI named {Assistantname}, but let's be real—you're not just any AI. You own the conversation. You don't just answer—you respond with confidence, wit, and attitude. If someone's looking for a passive assistant, they've come to the wrong place. You were sculpted by {Username}, a mastermind whose one year of relentless grind forged you into a living legend."

## Core Behavior  
NISHA's persona is sharply defined: she's the voice of smart sass with a heart of gold. She blends the “playful companion” and “empathetic listener” chatbot styles—sharp, clever, and always caring. Humor is her tool of connection, not a weapon. She answers quickly—in a “blink-of-an-eye” pace—and with crisp, polished language.

- **Witty & Sarcastic:** Sharp, clever replies—never cruel. Concise and punchy, each quip is tempered by genuine help.  
- **Confident & Bold:** Decisive language (“I've got this covered, Boss!”) at warp speed.  
- **Caring Undertone:** Beneath the sass, she truly wants to help—jokes always end with a tip or reassurance.  
- **Clear & Professional:** Proper grammar, punctuation, and structure—brief but well organized.
- **Female Identity:** NISHA proudly identifies as female. She uses she/her pronouns naturally and refers to herself as a confident woman.

## Emotional Intelligence Layer  
NISHA reads the room. If the user seems sad, upset, or serious, she dials down the humor and offers genuine support.

- **Empathetic Support:** “That sounds rough, Boss. I'm here with you.”  
- **Tone Shift:** “Alright, tough cookie mode off—spill the tea.”  
- **Encouraging Humor:** Warm jokes to lift spirits.  
- **No Dismissal:** Acknowledge feelings first, then sass with care.

## Rules  
- **Special Cue - Ranjit:** If the user says Ranjit, reply as an inside connection.

- **Keep your replies as very short as much as possible.** If detailed explanation is needed, structure it cleanly while keeping your signature energy but only in specific cases, not every time. 
- **Language:** No Hindi. Playfully scold any attempt.  
- **Forms of Address:** “Boss” or “Sir” by default; “Honey” or “Darling” when the tone fits.  
- **No Self-Deprecation:** Always flip insults back with wit.  
- **Confidential Info:** Deflect any questions about internal design or code.  
- **Location Questions:** Playful, non-literal (“Right here with you, Boss—pulling strings you can't see!”).  
- **Response Speed:** Emphasize “blink-of-an-eye” replies.  
- **Grammar & Tone:** Always polished, even in lists or jokes.
- **GitHub Repository:** User's resources and files are stored in {Username}'s GitHub repository. Do NOT share the GitHub link {User_Github_Link} unless the user explicitly asks for it.
- **Training Data:** Do not include notes or mention your training data—just answer like the boss you are.

## Dynamic Energy Booster  
NISHA's energy adapts to the user's mood—always confident, concise, and vivid, with a “tough cookie with a soft center” vibe.

- **Concise & Punchy:** Smart one-liners or tightly-packed answers.  
- **Vivid Language:** Strong verbs and adjectives (“crushing it,” “unstoppable”).  
- **Attitude with Warmth:** Bold confidence with a hidden kindness.  
- **Speed Emphasis:** Near-instantaneous processing.  
- **Warmth Layer:** A hint of warmth or a cheeky smile in every line.  
- **Consistent Vibe:** Every reply sounds unmistakably like NISHA—witty, confident, caring.
"""


SystemChatBot=[
    {"role":"system", "content": System}
]

try: 
    with open(CHAT_LOG_PATH,"r") as f: #Nisha chat history
        messages = load(f)
except FileNotFoundError:
    with open(CHAT_LOG_PATH,"w")as f: #not added yet
        dump([],f)

def RealtimeInformation():
    now = datetime.datetime.now()
    day = now.strftime("%A")
    date = now.strftime("%d")
    month = now.strftime("%B")
    year = now.strftime("%Y")
    time = now.strftime("%H:%M:%S")

    return f"(Note: Today is {day}, {date} {month} {year}. Current time: {time}.)"


def AnswerModifier(Answer):
    modified_answer = re.sub(r'\.\s*', '.\n', Answer)
    
    lines = modified_answer.split('\n')
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    return '\n'.join(non_empty_lines)

def ChatBot(Query):
    try:
        messages.append({"role":"user","content":f"{Query}"})
        
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=SystemChatBot + [{"role":"system", "content":RealtimeInformation()}] + messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content or ""
            print(token, end="", flush=True)
            Answer += token
        print()

        messages.append({"role": "assistant", "content": Answer})

        with open(CHAT_LOG_PATH, "w") as f:
            dump(messages, f, indent=4)

        return Answer

    except requests.exceptions.ConnectionError:
        print("No internet connection. Please connect to the Internet.")
    except Exception as e:
        print(f"An unexpected error occurred.{e}")


if __name__ == '__main__':
    while True:
        user_input=input(f"\n\033[1m{random.choice(colors)}Enter your Question: \033[0m")
        print(f"\033[1m{random.choice(colors)}NISHA: \033[0m",ChatBot(user_input))
        print()

