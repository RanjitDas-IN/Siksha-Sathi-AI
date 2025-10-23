#!/usr/bin/env python3
import os
import json
from datetime import datetime, timezone
from openai import OpenAI

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = "openai/gpt-oss-20b"
LOGFILE = "conversation_log.json"
 

client = OpenAI(api_key="", base_url=BASE_URL)

data = """
{
  "id": "19a1168eb167f2e8",
  "threadId": "19a1168eb167f2e8",
  "from": "Ranjit Das <noreply@github.com>",
  "to": "udaysubba2004@gmail.com",
  "subject": "RanjitDas-IN invited you to RanjitDas-IN/Siksha-Sathi-AI",
  "date": "Thu, 23 Oct 2025 07:11:19 -0700",
  "body": "@RanjitDas-IN has invited you to collaborate on the RanjitDas-IN/Siksha-Sathi-AI repository.\r\n\r\n\r\nVisit https://github.com/RanjitDas-IN/Siksha-Sathi-AI/invitations to accept or decline this invitation.\r\n\r\nYou can also head over to https://github.com/RanjitDas-IN/Siksha-Sathi-AI to check out the repository or visit https://github.com/RanjitDas-IN to learn a bit more about @RanjitDas-IN.\r\n\r\nThis invitation will expire in 7 days.\r\n\r\nSome helpful tips:\r\n- If you get a 404 page, make sure you’re signed in as kisxo.\r\n- Too many emails from @RanjitDas-IN? You can block them by visiting\r\n  https://github.com/settings/blocked_users?block_user=RanjitDas-IN or report abuse at\r\n  https://github.com/contact/report-abuse?report=RanjitDas-IN\r\n\r\n---\r\nView it on GitHub:\r\nhttps://github.com/RanjitDas-IN/Siksha-Sathi-AI"
}
"""


input_prompt = f"""Simplify this raw email message:

{data}
"""

def call_model(prompt_text, model=MODEL, max_tokens=5560):
    messages = [
        {"role": "system", "content": "You are Siksha-Sathi-AI.Your task is to read an email and rewrite it in simple, clear English that even a small child can understand. Do not add introductions like 'Here's the summary' or 'This is what the email says'. Just output the simplified summary itself — nothing else. You naver reply in markdown format"},
        {"role": "user", "content": prompt_text},
    ]

    resp = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
    )

    assistant_text = getattr(resp, "output_text", None)
    if not assistant_text:
        try:
            parts = []
            for item in resp.output:
                if isinstance(item, dict) and "content" in item:
                    for c in item["content"]:
                        if isinstance(c, dict) and "text" in c:
                            parts.append(c["text"])
                elif isinstance(item, str):
                    parts.append(item)
            assistant_text = "\n".join(parts).strip()
        except Exception:
            assistant_text = str(resp)

    return assistant_text

def append_to_log(user_msg, assistant_msg, raw_email=None, logfile=LOGFILE):
    entry = {
        "id": datetime.now(timezone.utc).astimezone().isoformat(),
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "raw_email": raw_email,
    }
    if os.path.exists(logfile):
        try:
            with open(logfile, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(entry)
    with open(logfile, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

def main():
    # print("Sending prompt to model...")
    assistant_reply = call_model(input_prompt)
    print("\n--- Assistant reply ---\n")
    print(assistant_reply)
    append_to_log(user_msg=input_prompt, assistant_msg=assistant_reply, raw_email=data)
    # print(f"\nConversation saved to {LOGFILE}")
    return assistant_reply

if __name__ == "__main__":
    main()
