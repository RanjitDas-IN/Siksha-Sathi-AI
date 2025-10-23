#!/usr/bin/env python3
"""
exam_todo.py

Generate a simple, actionable to-do list for any exam using provided exam examples.
Usage:
    export OPENAI_API_KEY="sk_...."
    python exam_todo.py "NEET"

If no exam name is passed as argv, the script will prompt you.
"""

import os
import sys
import json
import re
from datetime import datetime, timezone
from openai import OpenAI   # keep consistent with your environment; adjust if needed

# Config
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = "openai/gpt-oss-20b"
LOGFILE = "conversation_log.json"
 

client = OpenAI(api_key="", base_url=BASE_URL)
LOGFILE = "exam_todo_log.json"
MAX_TOKENS = 700


# Example data (replace or load from file). Keep it as a single string as you provided.
data = """ 

  {
    "exam_name": "RRB JE (Railway Recruitment Board Junior Engineer)",
    "exam_eligibility": "Candidates must have a diploma or degree in engineering in relevant branches. Age should be 18–33 years, with relaxation for reserved categories. Medical and physical standards must be met as per RRB requirements.",
    "recomended_topics": "Syllabus includes Technical subjects relevant to the engineering branch, General Intelligence and Reasoning, General Awareness, and Arithmetic. Core engineering concepts, problem-solving, and technical aptitude are key.",
    "exam_details": "Selection involves a Computer-Based Test (CBT), followed by Document Verification and Medical Examination. The CBT consists of objective-type questions covering technical knowledge, general awareness, reasoning, and numerical ability.",
    "career_scope": "Qualified candidates join as Junior Engineers in Indian Railways in departments like Civil, Mechanical, Electrical, and Electronics. Career offers structured promotions, technical and administrative responsibilities, transfers across India, and long-term benefits including pension."
  },

  {
    "exam_name": "CAT (Common Admission Test)",
    "exam_eligibility": "Candidates must hold a bachelor's degree with at least 50% marks (45% for SC/ST/PwD). Final-year students can also apply, provided they complete their degree before the admission process. There is no age limit, and the number of attempts is not restricted.",
    "recomended_topics": "Syllabus covers Quantitative Ability, Data Interpretation & Logical Reasoning, Verbal Ability & Reading Comprehension. Focus on algebra, arithmetic, geometry, critical reasoning, data interpretation, and comprehension passages.",
    "exam_details": "CAT is a computer-based test of 3 hours, divided into three sections: Verbal Ability & Reading Comprehension, Data Interpretation & Logical Reasoning, and Quantitative Ability. Each section is timed individually. Scores are normalized, and percentile ranking determines admission eligibility.",
    "career_scope": "Qualifying CAT allows admission to IIMs and other top business schools for MBA/PGDM programs. Career opportunities include management consulting, finance, marketing, operations, HR, entrepreneurship, and leadership roles in corporate sectors."
  },
  {
    "exam_name": "XAT (Xavier Aptitude Test)",
    "exam_eligibility": "Candidates must have a bachelor's degree in any discipline with a minimum of 50% marks. Final-year students are eligible. There is no age restriction, and attempts are not limited.",
    "recomended_topics": "Syllabus includes Verbal and Logical Ability, Decision Making, Quantitative Ability & Data Interpretation, and General Knowledge. Emphasis on analytical reasoning, problem-solving, comprehension, and business awareness.",
    "exam_details": "XAT is a computer-based test of 3 hours with multiple-choice questions and a few subjective questions in Decision Making. Sections include Verbal & Logical Ability, Decision Making, Quantitative Ability & Data Interpretation, and General Knowledge. Some universities consider essay writing or group discussion in selection.",
    "career_scope": "Qualifying XAT allows admission to XLRI and other associated B-schools. Graduates can pursue careers in management, consulting, finance, marketing, operations, and HR in India and globally."
  },
  {
    "exam_name": "MAT (Management Aptitude Test)",
    "exam_eligibility": "Candidates must hold a bachelor’s degree in any discipline or be in the final year of graduation. There is no upper age limit, and multiple attempts are allowed. MAT scores are accepted by over 600 B-schools in India.",
    "recomended_topics": "Syllabus includes Language Comprehension, Mathematical Skills, Data Analysis & Sufficiency, Intelligence & Critical Reasoning, and General Knowledge. Focus on quantitative aptitude, verbal ability, logical reasoning, and current affairs.",
    "exam_details": "MAT is a 2.5-hour paper-based or computer-based test with 200 multiple-choice questions. Each section carries 40 questions, and there is no negative marking. Scores are valid for one year and sent to selected B-schools.",
    "career_scope": "Qualifying MAT allows admission to numerous MBA/PGDM programs in India. Career paths include business management, corporate strategy, finance, marketing, operations, human resources, and entrepreneurship."
  },
  {
    "exam_name": "SNAP (Symbiosis National Aptitude Test)",
    "exam_eligibility": "Candidates must have a bachelor's degree with at least 50% marks (45% for reserved categories). Final-year students can also apply. There is no age limit or restriction on attempts.",
    "recomended_topics": "Syllabus covers General English, Quantitative, Data Interpretation & Data Sufficiency, Analytical & Logical Reasoning, and Current Affairs & General Knowledge. Focus on comprehension, reasoning, and analytical skills.",
    "exam_details": "SNAP is a computer-based test of 60 minutes, consisting of 60 multiple-choice questions. Each correct answer carries 1 mark, and 0.25 marks are deducted for incorrect answers. Some institutes may also consider group exercises or personal interviews.",
    "career_scope": "Qualifying SNAP allows admission to Symbiosis institutes for MBA/PGDM programs. Career options include management consulting, marketing, finance, operations, human resources, and entrepreneurship."
  }
"""

# ----------------------
# Helpers
# ----------------------
def parse_exam_data(raw: str):
    """
    Try to parse the provided raw exam data into a list of dicts.
    Accepts either a JSON array string, or a sequence of JSON objects (wrap with []).
    Falls back to regex-based object extraction if necessary.
    """
    text = raw.strip()
    # quick attempt: try json.loads directly
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    # If it looks like multiple objects without surrounding [], try wrapping
    try:
        wrapped = "[" + text + "]"
        parsed = json.loads(wrapped)
        return parsed
    except Exception:
        pass

    # fallback: find {...} blocks and json.loads each
    objects = re.findall(r"\{(?:[^{}]|(?R))*\}", text, flags=re.S)
    results = []
    for obj_text in objects:
        try:
            results.append(json.loads(obj_text))
        except Exception:
            # try to fix common issues like trailing commas
            cleaned = re.sub(r",\s*}", "}", obj_text)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                results.append(json.loads(cleaned))
            except Exception:
                continue
    return results


def find_exam_entry(exams, name):
    """
    Find an exam entry in parsed exams by matching name case-insensitively.
    If exact not found, try partial match.
    """
    if not exams:
        return None
    target = (name or "").strip().lower()
    for e in exams:
        if not isinstance(e, dict):
            continue
        if "exam_name" in e and isinstance(e["exam_name"], str):
            if e["exam_name"].strip().lower() == target:
                return e
    # partial match
    for e in exams:
        if "exam_name" in e and isinstance(e["exam_name"], str):
            if target in e["exam_name"].strip().lower():
                return e
    return None


def build_prompts(exam_data_str, exam_name):
    """
    Construct system and user prompts using the dynamic template approach.
    """
    system_prompt = """You are ExamPlanner, a smart assistant that generates actionable to-do lists for any exam.
Instructions:
- Use the exam entries the user provides as reference examples.
- Output ONLY a direct, plain-language to-do list.
- Keep it simple, numbered, and grouped in sections a student can follow.
- Sections to include (if relevant): Eligibility checklist, Syllabus breakdown, Daily/Weekly study plan, Mock tests & evaluation, Revision & retention, Resources, Final checklist.
- If the requested exam is NOT in the provided examples, infer reasonable steps/topics based on typical patterns of that exam type (e.g., engineering, medical, management).
- Include approximate timing suggestions for study blocks.
- Do NOT include intros, summaries, JSON, or extra commentary.
"""

    user_prompt = f"""
You are given example exam data below. Using those examples, produce a simple, actionable to-do list for: {exam_name}

Example data:
{exam_data_str}

Goal: Generate a concise, practical to-do list that a student (or a child) can follow. Start directly with the tasks, grouped into sections as instructed by the system prompt. Use numbered items and short lines. If specific details for the exam are missing, infer sensible and commonly used steps for that exam type.
"""
    return system_prompt, user_prompt


def call_model(system_prompt, user_prompt, model=MODEL, max_tokens=2000):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
    )

    # Try multiple extraction paths to get plain text
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    # Fallback: look inside resp.output
    output_texts = []
    for item in getattr(resp, "output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and "text" in content:
                output_texts.append(content["text"])
            elif content.get("type") == "reasoning_text" and "text" in content:
                output_texts.append(content["text"])
            elif "text" in content:
                output_texts.append(content["text"])

    return "\n".join(output_texts).strip()



def append_log(entry, logfile=LOGFILE):
    existing = []
    if os.path.exists(logfile):
        try:
            with open(logfile, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
        except Exception:
            existing = []
    existing.append(entry)
    with open(logfile, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ----------------------
# Main flow
# ----------------------
def generate_todo_for_exam(exam_name, raw_data):
    parsed = parse_exam_data(raw_data)
    matched = find_exam_entry(parsed, exam_name)
    system_prompt, user_prompt = build_prompts(raw_data, exam_name)

    # Add a short hint to user_prompt about whether exam matched or not
    if matched:
        user_prompt += f"\n\nNote: Use this entry as primary reference: {json.dumps(matched, ensure_ascii=False)}"
    else:
        user_prompt += "\n\nNote: No exact match found in examples; infer common syllabus/topics and steps relevant to the exam type."

    assistant_text = call_model(system_prompt, user_prompt)
    # ensure we don't accidentally echo the 'Here is' preamble — trim a few common phrases
    assistant_text = re.sub(r"^\s*(Here('|’)s|Here is|Summary:|Here's a summary of).*?(\n\s*)", "", assistant_text, flags=re.I|re.S)
    return assistant_text


def main():
    if len(sys.argv) >= 2:
        exam_name = " ".join(sys.argv[1:]).strip()
    else:
        exam_name = input("Enter exam name (e.g., NEET): ").strip()
    if not exam_name:
        print("No exam name provided. Exiting.")
        sys.exit(1)

    print(f"Generating to-do list for: {exam_name} ...\n")
    result = generate_todo_for_exam(exam_name, data)
    print(result)
    # Log for your records
    entry = {
        "id": datetime.now(timezone.utc).isoformat(),
        "exam_name": exam_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "todo_text": result,
    }
    append_log(entry)
    print(f"\nSaved to {LOGFILE}")


if __name__ == "__main__":
    main()
