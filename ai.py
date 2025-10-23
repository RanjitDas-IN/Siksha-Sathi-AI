#!/usr/bin/env python3
"""
Realtime token-by-token streaming with realtime log writes and an print_chunk hook
you can replace with a websocket/SSE sender.

- Streams tokens as `decoded_chunk` (the streaming boundary).
- Writes token chunks to LOG_FILE as they arrive (mimics uploaded script).
- Appends full user/assistant messages to LOG_FILE at end of turn.
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


data = """
[


  {
    "exam_name": "NIFT Entrance Exam (National Institute of Fashion Technology)",
    "exam_eligibility": "Candidates must have completed 10+2 or equivalent examination for undergraduate programs. For postgraduate programs, a bachelor’s degree in any discipline is required. There is no age limit, and multiple attempts are allowed.",
    "recomended_topics": "Syllabus includes Quantitative Ability, Communication Ability, English Comprehension, Analytical Ability, General Knowledge & Current Affairs, and Creative Ability Test (CAT) involving design, visualization, and drawing skills. Emphasis is on creativity, aesthetics, and design thinking.",
    "exam_details": "NIFT Entrance Exam consists of a Creative Ability Test (CAT) and General Ability Test (GAT). For PG programs, a Situation Test may be conducted to assess practical skills. CAT evaluates drawing, visualization, and innovation, while GAT covers general aptitude. Each stage is scored, and combined scores determine selection.",
    "career_scope": "Qualifying NIFT allows admission to fashion design, textile design, accessory design, and management programs. Career opportunities include fashion designing, textile industry, product design, visual merchandising, fashion marketing, and entrepreneurship."
  },
  {
    "exam_name": "NID DAT (National Institute of Design – Design Aptitude Test)",
    "exam_eligibility": "For undergraduate programs, candidates must have completed 10+2 or equivalent. For postgraduate programs, a bachelor’s degree in any discipline is required. There is no age limit, and multiple attempts are allowed.",
    "recomended_topics": "Syllabus includes Visualization Skills, Design Thinking, Creativity, Observation, and Problem-Solving. Emphasis on sketching, innovation, and conceptual understanding of design.",
    "exam_details": "NID DAT consists of a Design Aptitude Test (Prelims and Mains). Prelims assess drawing, design awareness, and creativity, while Mains include studio tests, personal interviews, and portfolio reviews. Scores from both rounds determine admission.",
    "career_scope": "Qualifying NID DAT allows admission to B.Des and M.Des programs. Career opportunities include industrial design, product design, UX/UI design, interior design, graphic design, and creative consultancy."
  },
  {
    "exam_name": "UCEED (Undergraduate Common Entrance Examination for Design)",
    "exam_eligibility": "Candidates must have passed 10+2 or equivalent with at least 50% marks. Final-year students can also apply. There is no age restriction, and multiple attempts are permitted.",
    "recomended_topics": "Syllabus includes Visualization and Spatial Ability, Observation and Design Sensitivity, Environmental and Social Awareness, Analytical and Logical Reasoning, Language, and Creativity. Focus on conceptual thinking and problem-solving.",
    "exam_details": "UCEED is a 3-hour computer-based test with three sections: Part A (objective questions on visual, verbal, and analytical ability), Part B (drawing section to test design and creativity), and Part C (evaluated for creativity and design aptitude).",
    "career_scope": "Qualifying UCEED allows admission to B.Des programs at IITs and other participating institutes. Career paths include industrial design, fashion and textile design, product design, UX/UI design, interior design, and creative consulting."
  },
  {
    "exam_name": "NCHM JEE (National Council for Hotel Management Joint Entrance Examination)",
    "exam_eligibility": "Candidates must have passed 10+2 or equivalent with English as a subject and at least 50% aggregate marks (45% for SC/ST). Age should be 17–25 years. Final-year students can apply, provided they complete the qualifying exam before admission.",
    "recomended_topics": "Syllabus includes English Language, Numerical Ability & Analytical Aptitude, Reasoning & Logical Deduction, General Knowledge & Current Affairs, and Hospitality & Hotel Management basics. Focus on comprehension, arithmetic, reasoning, and understanding of hospitality industry concepts.",
    "exam_details": "NCHM JEE is a computer-based test of 3 hours with 200 multiple-choice questions. Each correct answer carries 4 marks, and 1 mark is deducted for wrong answers. Sections include English, numerical ability, reasoning, GK, and domain-specific questions on hospitality.",
    "career_scope": "Qualifying NCHM JEE allows admission to B.Sc programs in Hospitality & Hotel Administration at affiliated institutes. Career opportunities include hotel management, hospitality operations, food and beverage management, tourism, event management, and luxury hotel chains."
  },
{
    "exam_name": "CTET (Central Teacher Eligibility Test)",
    "exam_eligibility": "Candidates must have passed Senior Secondary (or equivalent) with at least 50% marks and a 2-year Diploma in Elementary Education (D.El.Ed) or B.Ed. For general category, minimum marks are 50%, with relaxation for SC/ST/OBC/PwD. There is no upper age limit.",
    "recomended_topics": "Syllabus includes Child Development & Pedagogy, Language I & II, Mathematics, Environmental Studies, and Teaching Aptitude. Emphasis on understanding educational psychology, pedagogy, and subject knowledge for teaching classes I–VIII.",
    "exam_details": "CTET consists of two papers: Paper I for teaching classes I–V and Paper II for teaching classes VI–VIII. Each paper has 150 multiple-choice questions, 1 mark each, with no negative marking. Duration is 2 hours 30 minutes per paper.",
    "career_scope": "Qualifying CTET is mandatory for appointment as teachers in central government schools (KVS, NVS, and others). It enhances eligibility for state government and private schools and opens opportunities for teacher training, educational consultancy, and academic leadership roles."
  },
  {
    "exam_name": "TET (Teacher Eligibility Test – State Level)",
    "exam_eligibility": "Candidates must have passed Senior Secondary or equivalent with a diploma in elementary education or B.Ed, depending on the state-specific criteria. Minimum marks and age limits vary by state.",
    "recomended_topics": "Syllabus includes Child Development & Pedagogy, Language, Mathematics, Environmental Studies, and Teaching Aptitude. Focus is on pedagogy, subject knowledge, and classroom management.",
    "exam_details": "TET is conducted by individual states and generally consists of Paper I (for primary teachers) and Paper II (for upper primary teachers). Each paper includes multiple-choice questions, usually 150 questions, with a duration of 2.5 hours. Negative marking varies by state.",
    "career_scope": "Qualifying TET allows recruitment as teachers in state government schools for primary and upper primary levels. It also enhances eligibility for private schools and teaching roles in educational institutions."
  },
  {
    "exam_name": "NIOS Teacher Training Programs (NTT/CTT)",
    "exam_eligibility": "Candidates must have completed secondary or senior secondary education and meet the specific eligibility for the National Institute of Open Schooling teacher training programs. Age criteria may apply as per program guidelines.",
    "recomended_topics": "Syllabus includes Educational Psychology, Pedagogy, Teaching Methodologies, Communication Skills, and Subject Knowledge. Focus on practical teaching skills, lesson planning, and instructional design.",
    "exam_details": "Assessment includes written examinations, practical teaching demonstrations, and portfolio submissions. The duration and structure depend on the specific teacher training course.",
    "career_scope": "Completing NIOS teacher training programs enables candidates to work as trained teachers in open schooling, special education, and other government and private teaching institutions. It also opens opportunities in curriculum design and teacher training roles."
  },
  {
    "exam_name": "Uttarakhand/State Level School Teacher Recruitment Exams",
    "exam_eligibility": "Candidates must hold a bachelor’s degree in education (B.Ed) or equivalent, along with state-specific qualifications. Minimum marks and age limits are as per state government rules.",
    "recomended_topics": "Syllabus includes General Knowledge, Language Proficiency, Pedagogy, Subject-Specific Knowledge, and Teaching Aptitude. Emphasis on classroom management, lesson planning, and state curriculum awareness.",
    "exam_details": "State-level teacher recruitment exams are usually written objective-type tests, sometimes followed by interviews or teaching demonstrations. Duration is 2–3 hours, with 100–150 multiple-choice questions.",
    "career_scope": "Qualifying these exams allows recruitment as teachers in government schools within the state. It provides career stability, promotions, training opportunities, and eligibility for leadership positions in school administration."
  },
  {
    "exam_name": "NUEPA/DIET Teacher Recruitment Exams",
    "exam_eligibility": "Candidates must have completed graduation and teacher training programs recognized by the state or central government. Age and experience criteria vary by state.",
    "recomended_topics": "Syllabus covers Child Development, Pedagogy, Educational Psychology, Subject Knowledge, Language Proficiency, and Teaching Methods. Focus on inclusive education, teaching methodologies, and assessment techniques.",
    "exam_details": "Exams usually include written tests (objective and descriptive), teaching demonstrations, and interviews. The structure and duration depend on the recruiting body and state policies.",
    "career_scope": "Successful candidates are recruited as teachers in government-run DIETs (District Institutes of Education and Training), central schools, or state education departments. It provides opportunities for professional growth, specialized teaching roles, and positions in educational administration."
  },
 {
    "exam_name": "GMAT (Graduate Management Admission Test)",
    "exam_eligibility": "Candidates must hold a bachelor’s degree or equivalent in any discipline. There is no age limit, and candidates can take the exam multiple times. Final-year students can also apply.",
    "recomended_topics": "Syllabus includes Analytical Writing Assessment, Integrated Reasoning, Quantitative Reasoning, and Verbal Reasoning. Emphasis is on problem-solving, data interpretation, critical reasoning, and grammar.",
    "exam_details": "GMAT is a computer-adaptive test of approximately 3.5 hours. It consists of four sections: Analytical Writing, Integrated Reasoning, Quantitative, and Verbal. Scores range from 200–800, and each section has individual scoring.",
    "career_scope": "Qualifying GMAT allows admission to international business schools for MBA and specialized management programs. Career opportunities include management consulting, finance, marketing, operations, and leadership roles globally."
  },
  {
    "exam_name": "GRE (Graduate Record Examination)",
    "exam_eligibility": "Candidates must have a bachelor’s degree or be in the final year of undergraduate study. There is no age limit, and multiple attempts are allowed. GRE is mainly for admission to graduate programs abroad.",
    "recomended_topics": "Syllabus includes Verbal Reasoning, Quantitative Reasoning, and Analytical Writing. Focus on vocabulary, reading comprehension, algebra, geometry, data analysis, critical thinking, and essay writing.",
    "exam_details": "GRE is a computer-based or paper-based test lasting around 3 hours 45 minutes. It has three sections: Verbal Reasoning (two sections), Quantitative Reasoning (two sections), and Analytical Writing (two essays). Scoring ranges from 130–170 for Verbal and Quantitative and 0–6 for Analytical Writing.",
    "career_scope": "Qualifying GRE is required for admission to master’s or doctoral programs abroad, particularly in the US, Canada, and Europe. Careers include research, academia, engineering, business analytics, and global professional opportunities."
  },
  {
    "exam_name": "IELTS (International English Language Testing System)",
    "exam_eligibility": "There are no formal eligibility requirements. It is open to anyone who wants to study, work, or migrate to English-speaking countries. Candidates should have sufficient English proficiency to attempt the test.",
    "recomended_topics": "Syllabus includes Listening, Reading, Writing, and Speaking. Focus on comprehension, vocabulary, grammar, pronunciation, essay writing, and communication skills.",
    "exam_details": "IELTS is a 2 hour 45 minute test, consisting of four modules: Listening (40 minutes), Reading (60 minutes), Writing (60 minutes), and Speaking (11–14 minutes). Scores are reported on a 9-band scale for each module and overall.",
    "career_scope": "IELTS scores are used for university admissions, professional registration, and immigration purposes in countries like the UK, Australia, Canada, and New Zealand. It enhances global study and work opportunities."
  },
  {
    "exam_name": "TOEFL (Test of English as a Foreign Language)",
    "exam_eligibility": "There is no formal eligibility; it is open to anyone seeking to study in English-speaking universities. Candidates should have intermediate to advanced English proficiency.",
    "recomended_topics": "Syllabus covers Reading, Listening, Speaking, and Writing. Focus is on academic English, comprehension, essay writing, grammar, vocabulary, and verbal communication.",
    "exam_details": "TOEFL iBT is a computer-based test lasting about 3 hours, covering Reading, Listening, Speaking, and Writing sections. Each section is scored individually, with a total score range of 0–120.",
    "career_scope": "TOEFL scores are required for admission to universities abroad, particularly in the US and Canada. They support academic and professional pursuits in international settings and scholarship opportunities."
  },
  {
    "exam_name": "IELTS UKVI / PTE Academic",
    "exam_eligibility": "Open to candidates aiming for study or work visas in English-speaking countries. No formal age or education requirements, but English proficiency is necessary.",
    "recomended_topics": "Syllabus is similar to IELTS, including Listening, Reading, Writing, and Speaking. PTE Academic additionally evaluates integrated skills and English fluency.",
    "exam_details": "IELTS UKVI and PTE Academic are computer-based tests lasting 2–3 hours. Scores are reported on band scores (IELTS) or scaled scores (PTE). Both assess listening, reading, speaking, and writing abilities.",
    "career_scope": "These exams are required for immigration, student visas, and professional registration in countries like the UK, Australia, and New Zealand. High scores facilitate admission, employment, and global mobility."
  },
{
    "exam_name": "RBI Grade B – Reserve Bank of India Grade B Officer Exam",
    "exam_eligibility": "Candidates must be between 21 and 30 years old as of September 1, 2025. A minimum of 60% marks in Bachelor's degree or equivalent (50% for SC/ST/PwD) is required. For DEPR and DSIM posts, a Master's degree in Economics or Statistics is mandatory.",
    "recommended_topics": "For Phase 1: General Awareness, Reasoning Ability, English Language, Quantitative Aptitude. For Phase 2: Economic and Social Issues, Finance and Management, English (Writing Skills).",
    "exam_details": "The RBI Grade B exam consists of three phases: Phase 1 (Objective, 200 questions, 2 hours), Phase 2 (Objective & Descriptive, 300 marks, 5.5 hours), and an Interview (75 marks). Phase 1 includes General Awareness, Reasoning, English, and Quantitative Aptitude. Phase 2 includes Economic and Social Issues, Finance and Management, and English Writing Skills.",
    "career_scope": "Successful candidates are appointed as Officers in Grade B (DR) in the Reserve Bank of India. They can work in various departments like DEPR (Department of Economic and Policy Research), DSIM (Department of Statistics and Information Management), and others, with opportunities for career advancement in India's central banking institution."
  },
  {
    "exam_name": "IPMAT – Integrated Programme in Management Aptitude Test",
    "exam_eligibility": "Candidates should have passed Standard XII/HSC or equivalent examination in 2023, 2024, or appearing in 2025. For IIM Indore, candidates should be born on or after August 1, 2005, with a relaxation of 5 years for SC/ST/PwD categories.",
    "recommended_topics": "Quantitative Ability: Algebra, Arithmetic, Number Systems, Probability. Verbal Ability: Reading Comprehension, Vocabulary, Grammar. Logical Reasoning: Analytical Reasoning, Puzzles, Series.",
    "exam_details": "The IPMAT exam assesses candidates' aptitude in areas such as Quantitative Ability, Verbal Ability, and Logical Reasoning. It typically consists of multiple-choice questions and is conducted online.",
    "career_scope": "Candidates who clear IPMAT gain direct admission to the five-year Integrated Programme in Management at IIM Indore or Rohtak, leading to a Bachelor's degree followed by an MBA, preparing them for leadership roles in business and management."
  },
  {
    "exam_name": "UCEED – Undergraduate Common Entrance Examination for Design",
    "exam_eligibility": "Candidates should have passed Class XII (or equivalent) in the year 2025 or are appearing in the year 2026, in any stream (Science, Commerce, or Arts & Humanities) for the first time. Only those students who have passed Class XII (or equivalent) in the year 2025 or are appearing in the year 2026 are eligible to appear for UCEED 2026.",
    "recommended_topics": "Visual Perception, Spatial Ability, Environmental and Social Awareness, Analytical and Logical Reasoning, Design Thinking, Drawing Skills.",
    "exam_details": "UCEED consists of two parts: Part A (Computer-Based Test) and Part B (Sketching & Drawing Test). Part A includes Numerical Answer Type (NAT) Questions, Multiple-Select Questions (MSQ), and Multiple-Choice Questions (MCQ). Part B includes one subjective question (sketching-based) to be answered on paper.",
    "career_scope": "UCEED-qualified candidates are eligible for admission to the Bachelor of Design (B.Des) programs at IIT Bombay, IIT Guwahati, IIT Hyderabad, and IIITDM Jabalpur, leading to careers in product design, user experience, animation, and other design-related fields."
  },
  {
    "exam_name": "NCHMCT JEE – National Council for Hotel Management Joint Entrance Examination",
    "exam_eligibility": "Candidates should have passed the 10+2 examination from a recognized board with English as one of the subjects. Candidates appearing in the qualifying examination can also apply.",
    "recommended_topics": "Numerical Ability and Analytical Aptitude, Reasoning and Logical Deduction, General Knowledge and Current Affairs, English Language, Aptitude for the Service Sector.",
    "exam_details": "NCHMCT JEE is an objective-type exam with one question paper consisting of 200 questions. The total duration is 3 hours. The exam is conducted in CBT mode, and the medium of examination is English and Hindi.",
    "career_scope": "Candidates who qualify NCHMCT JEE can pursue a Bachelor's degree in Hotel Management and Catering Technology, leading to careers in hotel and hospitality management, food and beverage services, front office operations, and event management."
  },
{
    "exam_name": "UGAT – Under Graduate Aptitude Test",
    "exam_eligibility": "Candidates must have cleared 10+2 or equivalent examinations with at least 50% marks from a recognized board. Those candidates who are waiting for their result or appearing for the 10+2 examination are also eligible to apply.",
    "recommended_topics": "English Language: Verbal Reasoning, Sentence completion, Fill In the Blanks, One word substitution, Contextual usage, Vocabulary, Sentence correction, Idioms, Analogies, Different usage of the same word, Jumbled paragraphs. Numerical and Data Analysis: Geometry, Work and time, Number system.",
    "exam_details": "UGAT is a standardized test administered by AIMA for various undergraduate programs such as Integrated MBA (IMBA), BBA, BCA, BHM, B.Com, etc. The exam consists of multiple-choice questions covering English Language, Numerical and Data Analysis, Reasoning and Intelligence, and General Knowledge. The total duration is 2 hours, and the exam is conducted in both pen-and-paper and internet-based modes.",
    "career_scope": "Successful candidates can pursue undergraduate programs in management, commerce, and hospitality, leading to careers in business administration, information technology, hotel management, and related fields."
  },
  {
    "exam_name": "JMI Entrance Exam – Jamia Millia Islamia University Entrance Exam",
    "exam_eligibility": "Eligibility criteria vary by program. For undergraduate programs, candidates should have passed the senior secondary school certificate or an equivalent examination with a minimum of 50% marks in aggregate of best 5-papers or in the subject concerned.",
    "recommended_topics": "The syllabus includes subjects related to the specific program applied for, such as English, Mathematics, Science, and General Knowledge. The exam may consist of both objective and subjective questions, depending on the program.",
    "exam_details": "The JMI Entrance Exam is conducted annually for various undergraduate and postgraduate programs offered at Jamia Millia Islamia University. The exam is conducted in offline mode at various exam centers. The question paper will contain objective/multiple-choice questions (MCQs) and subjective questions in some programs.",
    "career_scope": "Candidates who clear the JMI Entrance Exam can pursue programs in fields such as engineering, humanities, social sciences, law, and management, leading to diverse career opportunities in academia, research, and industry."
  },
  {
    "exam_name": "FTII JET – Film and Television Institute of India Joint Entrance Test",
    "exam_eligibility": "For PG Diploma Courses (Film Wing): Bachelor's Degree in any discipline. For PG Diploma Courses (TV Wing): Bachelor's Degree in any discipline. For Acting Course: 10+2 or equivalent. Final year students are also eligible to apply.",
    "recommended_topics": "The syllabus includes topics related to Arts & Crafts, Film & TV, Allied Arts, Literature, General Awareness, Environmental Awareness, and General Knowledge related to Art and culture and Media.",
    "exam_details": "FTII JET is conducted for admission to various postgraduate diploma programs in film and television. The exam consists of a written test and an interview. The written test includes questions on general awareness, arts and crafts, and media-related topics.",
    "career_scope": "Successful candidates can pursue careers in film and television production, direction, editing, cinematography, acting, and other related fields in the entertainment industry."
  },
  {
    "exam_name": "APSC CCE – Assam Public Service Commission Combined Competitive Examination",
    "exam_eligibility": "Candidates must be a permanent resident of Assam and should have a Bachelor's Degree from a recognized university. The age limit is generally between 21 and 38 years, with relaxations for reserved categories.",
    "recommended_topics": "The syllabus includes General Studies, Indian and World Geography, Indian History, Indian Polity, Economic and Social Development, General Mental Ability, and current affairs.",
    "exam_details": "APSC CCE consists of three stages: Preliminary Examination (Objective Type), Main Examination (Written & Interview Test). The Preliminary Examination consists of two objective-type General Studies question papers, each carrying 200 marks. The Main Examination includes written papers and an interview.",
    "career_scope": "Candidates who qualify APSC CCE are appointed to various administrative posts in the Assam Government, such as Assam Civil Services, Police Services, and other allied services."
  }

]"""


def print_chunk(chunk: str):
    print(chunk, end="", flush=True)

# ---------- Config ----------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # change as needed
LOG_FILE = "conversation_log.txt"
CHUNK_SIZE = 1
MAX_TOTAL_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
SLEEP_BETWEEN_TOKENS = 0.0

# ensure log exists
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w", encoding="utf-8").close()

def append_to_log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()

def log_message(role: str, content: str):
    append_to_log(f"{role.upper()}: {content}\n\n")

# ---------- Model / tokenizer loader ----------
def load_model_and_tokenizer(model_name: str, device: str = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device

# ---------- Streaming generator ----------
def stream_response_generator(
    conversation: list,
    model,
    tokenizer,
    device,
    chunk_size: int = CHUNK_SIZE,
    max_new_tokens: int = MAX_TOTAL_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    sleep_between_tokens: float = SLEEP_BETWEEN_TOKENS,
):
    """
    Yields decoded_chunk strings as they arrive.
    Caller is responsible for forwarding each chunk (e.g., ws.send(chunk)).
    """
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    generated_ids = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    total_generated = 0
    stop_early = False

    with torch.inference_mode():
        while total_generated < max_new_tokens and not stop_early:
            outputs = model.generate(
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
                max_new_tokens=chunk_size,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            start_idx = generated_ids.shape[-1]
            new_tokens = outputs[:, start_idx:]  # (1, new_len)

            if new_tokens.shape[-1] == 0:
                break

            # extend generated_ids and attention mask
            generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)
            new_mask = torch.ones((generated_attention_mask.shape[0], new_tokens.shape[-1]),
                                  dtype=generated_attention_mask.dtype, device=device)
            generated_attention_mask = torch.cat([generated_attention_mask, new_mask], dim=-1)

            # ===== streaming boundary =====
            decoded_chunk = tokenizer.decode(new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # yield the chunk so caller can forward it to frontend in realtime
            yield decoded_chunk
            # ==============================

            total_generated += new_tokens.shape[-1]

            # stop on EOS
            if (new_tokens == tokenizer.eos_token_id).any():
                stop_early = True
                break

            if sleep_between_tokens > 0:
                time.sleep(sleep_between_tokens)

# ---------- Convenience wrapper that accepts an print_chunk callback ----------
def stream_response_callback(
    conversation: list,
    model,
    tokenizer,
    device,
    print_chunk,
    chunk_size: int = CHUNK_SIZE,
    max_new_tokens: int = MAX_TOTAL_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    sleep_between_tokens: float = SLEEP_BETWEEN_TOKENS,
    realtime_token_log: bool = True,
    log_file_path: str = LOG_FILE,
):
    """
    Calls print_chunk(decoded_chunk) for every token-chunk produced.
    Also writes chunks to the log file in realtime if realtime_token_log=True.
    Returns the final assembled assistant string.
    """
    assistant_buffer = ""

    # optionally open log file once for speed
    token_log_file = None
    if realtime_token_log:
        token_log_file = open(log_file_path, "a", encoding="utf-8")

    try:
        for chunk in stream_response_generator(
            conversation,
            model,
            tokenizer,
            device,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            sleep_between_tokens=sleep_between_tokens,
        ):
            # forward to user-provided callback (e.g., websocket send)
            try:
                print_chunk(chunk)
            except Exception:
                # if frontend send fails, propagate after cleanup
                raise

            assistant_buffer += chunk

            # write chunk to log in realtime (no newline)
            if token_log_file is not None:
                token_log_file.write(chunk)
                token_log_file.flush()
    finally:
        if token_log_file is not None:
            token_log_file.write("\n\n")
            token_log_file.close()

    return assistant_buffer

# ---------- Main realtime loop (console example) ----------
def ai(user_input: str, rag_data: str = '', user_data: str='', chat_history: str=''):

    print(user_data)
    return
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    system_message = {"role": "system", "content": "You are a helpful assistant. Always respond clearly.", "custom_data": rag_data}
    conversation = [system_message]
    conversation.append({"role": "user", "content": user_input})

    # call streaming wrapper which will also append tokens to LOG_FILE realtime
    final_text = stream_response_callback(
        conversation,
        model,
        tokenizer,
        device,
        print_chunk=print_chunk,
        chunk_size=CHUNK_SIZE,
        max_new_tokens=MAX_TOTAL_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        sleep_between_tokens=SLEEP_BETWEEN_TOKENS,
        realtime_token_log=True,
        log_file_path=LOG_FILE,
    )


ai(user_input=f"help me generate a todo list for neet exam bellow is some contex ")

