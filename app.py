import os, json, re, sqlite3, asyncio, time
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
import anthropic
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Setup ─────────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
IMAGE_DIR  = Path("slide_images")
STATIC_DIR = Path("static")
DB_PATH    = "study.db"

for d in [UPLOAD_DIR, IMAGE_DIR, STATIC_DIR]:
    d.mkdir(exist_ok=True)

client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_KEY") or os.environ.get("ANTHROPIC_API_KEY")
)

app = FastAPI()
app.mount("/slide_images", StaticFiles(directory="slide_images"), name="slide_images")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Database ──────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decks (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            pdf_path         TEXT NOT NULL,
            created_at       TEXT DEFAULT (datetime('now')),
            total_slides     INTEGER DEFAULT 0,
            processed_slides INTEGER DEFAULT 0,
            status           TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS slides (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_id      INTEGER REFERENCES decks(id) ON DELETE CASCADE,
            slide_number INTEGER,
            text_content TEXT,
            image_path   TEXT
        );
        CREATE TABLE IF NOT EXISTS topics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            deck_id         INTEGER REFERENCES decks(id) ON DELETE CASCADE,
            name            TEXT NOT NULL,
            order_num       INTEGER DEFAULT 0,
            anchor_slide_id INTEGER REFERENCES slides(id),
            overview_json   TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS concepts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            slide_id    INTEGER REFERENCES slides(id) ON DELETE CASCADE,
            deck_id     INTEGER REFERENCES decks(id) ON DELETE CASCADE,
            topic_id    INTEGER REFERENCES topics(id),
            concept     TEXT,
            explanation TEXT,
            type        TEXT
        );
        CREATE TABLE IF NOT EXISTS questions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            concept_id          INTEGER REFERENCES concepts(id) ON DELETE CASCADE,
            slide_id            INTEGER REFERENCES slides(id) ON DELETE CASCADE,
            deck_id             INTEGER REFERENCES decks(id) ON DELETE CASCADE,
            mcq_json            TEXT,
            short_answer_json   TEXT,
            applied_json        TEXT
        );
        CREATE TABLE IF NOT EXISTS mastery (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id    INTEGER REFERENCES questions(id) ON DELETE CASCADE,
            question_type  TEXT,
            attempts       INTEGER DEFAULT 0,
            correct_streak INTEGER DEFAULT 0,
            avg_score      REAL DEFAULT 0.0,
            last_score     REAL DEFAULT 0.0,
            mastered       INTEGER DEFAULT 0,
            last_tested    TEXT,
            UNIQUE(question_id, question_type)
        );
        CREATE TABLE IF NOT EXISTS responses (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id    INTEGER,
            question_type  TEXT,
            attempt_number INTEGER,
            user_answer    TEXT,
            score          REAL,
            feedback_json  TEXT,
            answered_at    TEXT DEFAULT (datetime('now'))
        );
    """)
    # Migration: add topic_id to concepts if not present in older DBs
    try:
        conn.execute("ALTER TABLE concepts ADD COLUMN topic_id INTEGER REFERENCES topics(id)")
        conn.commit()
    except Exception:
        pass
    conn.commit()
    conn.close()

init_db()

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(raw: str) -> list[str]:
    lines = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and "CMPSC 311" not in line and not line.startswith("Devic"):
            lines.append(line)
    return lines

def parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

# ── Prompts ───────────────────────────────────────────────────────────────────
GROUP_PROMPT = """You are organizing a CMPSC 311 (Systems Programming) lecture slide deck into study groups.

Your job: group consecutive slides that cover the same topic, concept, or animated sequence into one unit.

Rules for grouping:
- Animation/build sequences: when the professor shows the same diagram/code building up over multiple slides → ONE group, anchor = the LAST slide (most complete state)
- Same topic, multiple slides: e.g., "The Heap" across 3 slides → ONE group
- Clear topic change: new header/title that introduces a different concept → new group
- Single standalone slide: fine as its own group

Action per group:
- "skip": title-only slides (course name + lecture topic + meme), pure administrative/logistics slides, office hours, agenda slides, slides that say "in this class we will..." or "in 473 you will..."
- "study": anything with definitions, code, mechanisms, diagrams, or facts that could appear on an exam

IMPORTANT: Every slide number must appear in exactly one group. Do not drop any slides.

Return ONLY valid JSON (no markdown fences):
{
  "groups": [
    {"slides": [1], "anchor": 1, "topic": "brief topic name", "action": "skip"},
    {"slides": [2, 3, 4], "anchor": 4, "topic": "Static vs automatic memory allocation", "action": "study"}
  ]
}"""

EXTRACTION_PROMPT = """You are in EXAM WRITER MODE for CMPSC 311 (Systems Programming / Intro to C and OS).

Analyze this lecture slide (image + extracted text) and extract EVERY discrete concept.

Rules:
- Atomic units: one fact = one item. If a bullet has 3 ideas, list 3 items.
- Define every term mentioned precisely.
- For code: what it does, why, and what mistakes students make.
- For diagrams and memory maps: describe the structure, spatial layout, and what each region represents.
- Do NOT skip small details — this professor tests exact wording.

Respond with ONLY valid JSON (no markdown fences):
{
  "concepts": [
    {
      "concept": "short descriptive name",
      "explanation": "complete, precise explanation a student could memorize verbatim",
      "type": "definition|mechanism|relationship|example|formula|code_behavior|visual_diagram"
    }
  ]
}"""

QUESTION_PROMPT = """You are writing exam questions for CMPSC 311 (Systems Programming / C).

The professor tests EXACT slide content: memorization, code reading, true/false embedded in MC.
Make 30% of questions tricky — targeting common student misconceptions.

For EACH concept below, generate all 3 question types.

Respond with ONLY valid JSON (no markdown fences):
{
  "questions": [
    {
      "concept_index": 0,
      "mcq": {
        "question": "",
        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
        "correct": "A",
        "explanation": "why correct answer is right AND why each wrong option is wrong"
      },
      "short_answer": {
        "question": "",
        "key_points": ["must mention this", "must mention this", "must mention this"],
        "model_answer": "complete expected answer",
        "explanation": ""
      },
      "applied": {
        "question": "scenario or code snippet to analyze — be specific and concrete",
        "answer_framework": "step-by-step expected reasoning",
        "explanation": ""
      }
    }
  ]
}"""

GRADING_PROMPT = """Grade this CMPSC 311 student answer. Be strict — they need exact mastery for the exam.

Question: {question}
Expected key points: {key_points}
Model answer: {model_answer}
Student answer (attempt {attempt}): {answer}

Scoring: full credit for each key point correctly addressed, deduct for vague/wrong answers.
If score < 8, generate 2 probing follow-up questions that target exactly what they missed.

Respond with ONLY valid JSON (no markdown fences):
{{
  "score": <integer 0-10>,
  "hit_points": ["key points the student got right"],
  "missing_points": ["key points missing or stated incorrectly"],
  "misconceptions": ["any wrong ideas present in their answer"],
  "feedback": "2-3 sentences of direct, targeted feedback",
  "follow_up_questions": ["follow-up 1", "follow-up 2"]
}}

follow_up_questions must be empty array [] if score >= 8."""

OVERVIEW_PROMPT = """You are writing a study guide entry for CMPSC 311 (Systems Programming).

Topic: {topic}

Concepts in this topic:
{concepts_text}

Slide text:
{slide_text}

Return ONLY valid JSON (no markdown fences):
{{
  "summary": "3-4 sentences explaining what this topic covers, how it works, and why it matters in systems programming",
  "exam_signals": ["a specific thing likely to appear on an exam", "another likely tested item", "a third one"],
  "key_items": ["key API call or rule with brief note, e.g. 'malloc(size_t n) — allocates n bytes on the heap, returns void* or NULL on failure'", "another key item"]
}}"""

# ── Difficulty mapping ─────────────────────────────────────────────────────────
DIFFICULTY_TO_TYPE = {'easy': 'mcq', 'medium': 'short_answer', 'tricky': 'applied'}
TYPE_TO_DIFFICULTY = {v: k for k, v in DIFFICULTY_TO_TYPE.items()}

# ── Pipeline (sync, runs in background thread) ────────────────────────────────
def group_slides_sync(slide_texts: list[dict]) -> list[dict]:
    all_nums    = [s["slide_number"] for s in slide_texts]
    slides_block = "\n\n".join(
        f"=== Slide {s['slide_number']} ===\n{s['text'] or '(no text)'}"
        for s in slide_texts
    )
    try:
        msg = _claude_with_retry(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content":
                f"{GROUP_PROMPT}\n\nTotal slides: {len(slide_texts)}\nSlide numbers: {all_nums}\n\n{slides_block}"
            }]
        )
        groups = parse_json_response(msg.content[0].text).get("groups", [])
        seen = {s for g in groups for s in g.get("slides", [])}
        for n in all_nums:
            if n not in seen:
                groups.append({"slides": [n], "anchor": n, "topic": f"Slide {n}", "action": "study"})
        return groups
    except Exception as e:
        print(f"  [!] Grouping error: {e} — defaulting to one group per slide")
        return [
            {"slides": [s["slide_number"]], "anchor": s["slide_number"],
             "topic": f"Slide {s['slide_number']}",
             "action": "skip" if s["slide_number"] == 1 else "study"}
            for s in slide_texts
        ]

def extract_concepts_sync(combined_text: list[str], image_path: str, group_label: str) -> list[dict]:
    try:
        import base64 as _b64
        with open(image_path, "rb") as f:
            img_b64 = _b64.b64encode(f.read()).decode()
        text_block = "\n".join(combined_text) or "(no extractable text)"
        context = f"Topic: {group_label}\n\nCombined text from this slide group:\n{text_block}"
        msg = _claude_with_retry(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": f"{EXTRACTION_PROMPT}\n\n{context}"}
            ]}]
        )
        return parse_json_response(msg.content[0].text).get("concepts", [])
    except Exception as e:
        print(f"  [!] Concept extraction error ({group_label}): {e}")
        return []

def _claude_with_retry(model, max_tokens, messages, retries=4, base_delay=5):
    """Call Claude API with exponential backoff on overload/rate-limit errors."""
    for attempt in range(retries):
        try:
            return client.messages.create(model=model, max_tokens=max_tokens, messages=messages)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"  [!] API overloaded (attempt {attempt+1}/{retries}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

def generate_questions_sync(concepts: list[dict]) -> list[dict]:
    if not concepts:
        return []
    all_questions = []
    BATCH = 3
    for i in range(0, len(concepts), BATCH):
        batch = concepts[i:i + BATCH]
        try:
            msg = _claude_with_retry(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content":
                    f"{QUESTION_PROMPT}\n\nConcepts:\n{json.dumps(batch, indent=2)}"
                }]
            )
            batch_qs = parse_json_response(msg.content[0].text).get("questions", [])
            for q in batch_qs:
                q["concept_index"] = i + q.get("concept_index", 0)
            all_questions.extend(batch_qs)
        except Exception as e:
            print(f"  [!] Question generation error (batch {i // BATCH + 1}): {e}")
            continue
    return all_questions

def generate_topic_overview_sync(topic_name: str, concepts: list[dict], combined_text: str) -> dict:
    concepts_text = "\n".join(
        f"- {c.get('concept', '')}: {c.get('explanation', '')}"
        for c in concepts[:15]
    )
    prompt = OVERVIEW_PROMPT.format(
        topic=topic_name,
        concepts_text=concepts_text[:3000],
        slide_text=combined_text[:2000]
    )
    try:
        msg = _claude_with_retry(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return parse_json_response(msg.content[0].text)
    except Exception as e:
        print(f"  [!] Overview error ({topic_name}): {e}")
        return {"summary": f"Topic: {topic_name}", "exam_signals": [], "key_items": []}

def _store_group_questions(conn, concepts: list[dict], anchor_slide_id: int, deck_id: int, topic_id: int = None):
    """Insert concepts + questions + mastery rows for one group."""
    concept_ids = []
    for c in concepts:
        cur = conn.execute(
            "INSERT INTO concepts (slide_id, deck_id, topic_id, concept, explanation, type) VALUES (?,?,?,?,?,?)",
            (anchor_slide_id, deck_id, topic_id,
             c.get("concept", ""), c.get("explanation", ""), c.get("type", "definition"))
        )
        concept_ids.append(cur.lastrowid)
    conn.commit()

    questions = generate_questions_sync(concepts)
    for q in questions:
        ci = min(q.get("concept_index", 0), len(concept_ids) - 1)
        cur = conn.execute(
            "INSERT INTO questions (concept_id, slide_id, deck_id, mcq_json, short_answer_json, applied_json) VALUES (?,?,?,?,?,?)",
            (concept_ids[ci], anchor_slide_id, deck_id,
             json.dumps(q["mcq"])          if q.get("mcq")          else None,
             json.dumps(q["short_answer"])  if q.get("short_answer") else None,
             json.dumps(q["applied"])       if q.get("applied")      else None)
        )
        q_id = cur.lastrowid
        for qt in ("mcq", "short_answer", "applied"):
            conn.execute("INSERT OR IGNORE INTO mastery (question_id, question_type) VALUES (?,?)", (q_id, qt))
    conn.commit()
    return len(questions)

def process_pdf_background(deck_id: int, pdf_path: str):
    conn = get_db()
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        conn.execute("UPDATE decks SET total_slides=?, status='processing' WHERE id=?", (total, deck_id))
        conn.commit()
        print(f"[Deck {deck_id}] Phase 1: Rendering {total} slides...")

        # ── Phase 1: Render all slides ─────────────────────────────────────────
        all_slide_data = []
        for i, page in enumerate(doc):
            slide_num = i + 1
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img_path = str(IMAGE_DIR / f"d{deck_id}_s{slide_num}.png")
            pix.save(img_path)
            raw_text = page.get_text("text")
            cleaned  = clean_text(raw_text)
            cur = conn.execute(
                "INSERT INTO slides (deck_id, slide_number, text_content, image_path) VALUES (?,?,?,?)",
                (deck_id, slide_num, "\n".join(cleaned), img_path)
            )
            all_slide_data.append({
                "slide_number": slide_num,
                "slide_id":     cur.lastrowid,
                "img_path":     img_path,
                "text_lines":   cleaned,
            })
        conn.commit()

        # ── Phase 2: Group all slides ──────────────────────────────────────────
        print(f"[Deck {deck_id}] Phase 2: Grouping slides by topic...")
        slide_texts = [{"slide_number": s["slide_number"], "text": "\n".join(s["text_lines"])}
                       for s in all_slide_data]
        groups = group_slides_sync(slide_texts)
        slide_lookup = {s["slide_number"]: s for s in all_slide_data}

        study_groups = [g for g in groups if g.get("action") == "study"]
        skip_groups  = [g for g in groups if g.get("action") != "study"]
        print(f"[Deck {deck_id}] {len(study_groups)} study groups, {len(skip_groups)} skipped")

        # ── Phase 3: Process each study group ─────────────────────────────────
        print(f"[Deck {deck_id}] Phase 3: Processing {len(study_groups)} groups...")
        for gi, group in enumerate(study_groups):
            anchor_num = group.get("anchor")
            topic      = group.get("topic", f"Group {gi+1}")
            slide_nums = group.get("slides", [anchor_num])

            if anchor_num not in slide_lookup:
                continue
            anchor = slide_lookup[anchor_num]

            # Combine text from all slides in group, deduped
            seen_lines, combined = set(), []
            for sn in slide_nums:
                for line in slide_lookup.get(sn, {}).get("text_lines", []):
                    if line not in seen_lines:
                        seen_lines.add(line)
                        combined.append(line)

            print(f"  Group {gi+1}/{len(study_groups)}: '{topic}' (slides {slide_nums})")
            concepts = extract_concepts_sync(combined, anchor["img_path"], topic)

            if concepts:
                # Generate overview (needs concepts for context)
                overview = generate_topic_overview_sync(topic, concepts, "\n".join(combined))

                # Create topic record
                topic_cur = conn.execute(
                    "INSERT INTO topics (deck_id, name, order_num, anchor_slide_id, overview_json) VALUES (?,?,?,?,?)",
                    (deck_id, topic, gi, anchor["slide_id"], json.dumps(overview))
                )
                topic_id = topic_cur.lastrowid
                conn.commit()

                # Store concepts (with topic_id) + questions
                n_q = _store_group_questions(conn, concepts, anchor["slide_id"], deck_id, topic_id)
                print(f"    → {len(concepts)} concepts, {n_q} question sets")

            conn.execute("UPDATE decks SET processed_slides=? WHERE id=?", (anchor_num, deck_id))
            conn.commit()

        conn.execute("UPDATE decks SET processed_slides=total_slides, status='done' WHERE id=?", (deck_id,))
        conn.commit()
        print(f"[Deck {deck_id}] Done.")
    except Exception as e:
        print(f"[Deck {deck_id}] Error: {e}")
        import traceback; traceback.print_exc()
        conn.execute("UPDATE decks SET status='error' WHERE id=?", (deck_id,))
        conn.commit()
    finally:
        conn.close()

# ── Study Logic ───────────────────────────────────────────────────────────────
def get_next_question(
    deck_id: int,
    topic_id: int = None,
    difficulty: str = None,
    target_slide: int = None
) -> Optional[dict]:
    conn = get_db()
    try:
        question_type = DIFFICULTY_TO_TYPE.get(difficulty) if difficulty else None

        where_parts = ["q.deck_id = ?", "m.mastered = 0"]
        params = [deck_id]

        if topic_id:
            where_parts.append("c.topic_id = ?")
            params.append(topic_id)

        if target_slide:
            where_parts.append("s.slide_number = ?")
            params.append(target_slide)

        if question_type:
            where_parts.append("m.question_type = ?")
            params.append(question_type)
            where_parts.append(f"q.{question_type}_json IS NOT NULL")
        else:
            where_parts.append("""(
                (m.question_type = 'mcq'          AND q.mcq_json IS NOT NULL) OR
                (m.question_type = 'short_answer' AND q.short_answer_json IS NOT NULL) OR
                (m.question_type = 'applied'      AND q.applied_json IS NOT NULL)
            )""")

        where_clause  = " AND ".join(where_parts)
        order_clause  = (
            "m.attempts ASC, s.slide_number ASC, m.avg_score ASC"
            if question_type else
            "CASE m.question_type WHEN 'mcq' THEN 1 WHEN 'short_answer' THEN 2 ELSE 3 END ASC, "
            "m.attempts ASC, s.slide_number ASC, m.avg_score ASC"
        )

        row = conn.execute(f"""
            SELECT
                q.id AS question_id,
                q.mcq_json, q.short_answer_json, q.applied_json,
                s.slide_number, s.image_path,
                c.concept, c.explanation, c.topic_id,
                m.question_type, m.attempts, m.avg_score, m.correct_streak, m.mastered
            FROM questions q
            JOIN slides s   ON q.slide_id  = s.id
            JOIN concepts c ON q.concept_id = c.id
            JOIN mastery m  ON m.question_id = q.id
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT 1
        """, params).fetchone()

        if not row:
            return None

        r  = dict(row)
        qt = r["question_type"]
        field  = {"mcq": "mcq_json", "short_answer": "short_answer_json", "applied": "applied_json"}[qt]
        q_data = json.loads(r[field]) if r[field] else None
        if not q_data:
            return None

        # Scope stats for progress display (filtered to same topic+difficulty if set)
        stat_parts  = ["q.deck_id = ?"]
        stat_params = [deck_id]
        if topic_id:
            stat_parts.append("c.topic_id = ?")
            stat_params.append(topic_id)
        if question_type:
            stat_parts.append("m.question_type = ?")
            stat_params.append(question_type)

        stats = dict(conn.execute(f"""
            SELECT COUNT(*) as total, COALESCE(SUM(m.mastered), 0) as mastered
            FROM mastery m
            JOIN questions q ON q.id = m.question_id
            JOIN concepts c  ON c.id = q.concept_id
            WHERE {' AND '.join(stat_parts)}
        """, stat_params).fetchone())

        img_name = Path(r["image_path"]).name if r["image_path"] else None
        return {
            "question_id":         r["question_id"],
            "question_type":       qt,
            "difficulty":          TYPE_TO_DIFFICULTY.get(qt, qt),
            "slide_number":        r["slide_number"],
            "image_url":           f"/slide_images/{img_name}" if img_name else None,
            "concept":             r["concept"],
            "concept_explanation": r["explanation"],
            "question_data":       q_data,
            "attempts":            r["attempts"],
            "avg_score":           r["avg_score"],
            "scope_total":         stats["total"],
            "scope_mastered":      stats["mastered"],
        }
    finally:
        conn.close()

def grade_answer_sync(question_data: dict, user_answer: str, attempt: int, question_type: str) -> dict:
    question_text = question_data.get("question", "")
    key_points    = question_data.get("key_points", [question_data.get("answer_framework", "")])
    model_answer  = question_data.get("model_answer", question_data.get("answer_framework", ""))
    prompt = GRADING_PROMPT.format(
        question=question_text,
        key_points=json.dumps(key_points),
        model_answer=model_answer,
        attempt=attempt,
        answer=user_answer,
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_json_response(msg.content[0].text)

def update_mastery_db(question_id: int, question_type: str, score: float):
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT attempts, correct_streak, avg_score FROM mastery WHERE question_id=? AND question_type=?",
            (question_id, question_type)
        ).fetchone()
        if not row:
            return
        new_attempts  = row["attempts"] + 1
        new_avg       = (row["avg_score"] * row["attempts"] + score) / new_attempts
        new_streak    = (row["correct_streak"] + 1) if score >= 8 else 0
        streak_needed = 1 if question_type == "mcq" else 2
        mastered      = 1 if (new_streak >= streak_needed and new_avg >= 7.0) else 0
        conn.execute("""
            UPDATE mastery
            SET attempts=?, correct_streak=?, avg_score=?, last_score=?,
                mastered=?, last_tested=datetime('now')
            WHERE question_id=? AND question_type=?
        """, (new_attempts, new_streak, new_avg, score, mastered, question_id, question_type))
        conn.commit()
    finally:
        conn.close()

# ── Reprocess (fixes slides with concepts but missing questions) ───────────────
def reprocess_questions_background(deck_id: int):
    conn = get_db()
    try:
        slides = conn.execute("""
            SELECT s.id, s.slide_number FROM slides s
            WHERE s.deck_id = ?
              AND EXISTS     (SELECT 1 FROM concepts  c WHERE c.slide_id = s.id)
              AND NOT EXISTS (SELECT 1 FROM questions q WHERE q.slide_id = s.id)
            ORDER BY s.slide_number
        """, (deck_id,)).fetchall()

        total = len(slides)
        print(f"[Reprocess deck {deck_id}] {total} slides need questions")
        conn.execute("UPDATE decks SET total_slides=?, processed_slides=0, status='processing' WHERE id=?",
                     (total, deck_id))
        conn.commit()

        for idx, slide in enumerate(slides):
            slide_id = slide["id"]
            # Look up topic for this slide
            topic_row = conn.execute(
                "SELECT id FROM topics WHERE anchor_slide_id=? AND deck_id=?", (slide_id, deck_id)
            ).fetchone()
            topic_id = topic_row["id"] if topic_row else None

            concepts_rows = conn.execute(
                "SELECT id, concept, explanation, type FROM concepts WHERE slide_id=? ORDER BY id", (slide_id,)
            ).fetchall()
            concepts    = [dict(r) for r in concepts_rows]
            concept_ids = [r["id"] for r in concepts_rows]

            questions = generate_questions_sync(concepts)
            for q in questions:
                ci = min(q.get("concept_index", 0), len(concept_ids) - 1)
                cur = conn.execute(
                    "INSERT INTO questions (concept_id, slide_id, deck_id, mcq_json, short_answer_json, applied_json) VALUES (?,?,?,?,?,?)",
                    (concept_ids[ci], slide_id, deck_id,
                     json.dumps(q["mcq"])          if q.get("mcq")          else None,
                     json.dumps(q["short_answer"])  if q.get("short_answer") else None,
                     json.dumps(q["applied"])       if q.get("applied")      else None)
                )
                q_id = cur.lastrowid
                for qt in ("mcq", "short_answer", "applied"):
                    conn.execute("INSERT OR IGNORE INTO mastery (question_id, question_type) VALUES (?,?)", (q_id, qt))
            conn.commit()
            conn.execute("UPDATE decks SET processed_slides=? WHERE id=?", (idx + 1, deck_id))
            conn.commit()
            print(f"  Slide {slide['slide_number']}: {len(questions)} question sets")

        conn.execute("UPDATE decks SET status='done' WHERE id=?", (deck_id,))
        conn.commit()
    except Exception as e:
        print(f"[Reprocess deck {deck_id}] Error: {e}")
        conn.execute("UPDATE decks SET status='error' WHERE id=?", (deck_id,))
        conn.commit()
    finally:
        conn.close()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")
    pdf_path = UPLOAD_DIR / file.filename
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO decks (name, pdf_path) VALUES (?,?)",
        (Path(file.filename).stem, str(pdf_path))
    )
    deck_id = cur.lastrowid
    conn.commit()
    conn.close()
    background_tasks.add_task(process_pdf_background, deck_id, str(pdf_path))
    return {"deck_id": deck_id, "name": Path(file.filename).stem}

@app.get("/decks")
async def list_decks():
    conn = get_db()
    rows = conn.execute("""
        SELECT d.*,
            COUNT(DISTINCT q.id)         AS total_questions,
            COALESCE(SUM(m.mastered), 0) AS mastered_questions
        FROM decks d
        LEFT JOIN questions q ON q.deck_id = d.id
        LEFT JOIN mastery m   ON m.question_id = q.id
        GROUP BY d.id
        ORDER BY d.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/deck/{deck_id}/status")
async def deck_status(deck_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM decks WHERE id=?", (deck_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Not found")
    return dict(row)

@app.get("/deck/{deck_id}/topics")
async def get_topics(deck_id: int):
    conn = get_db()
    rows = conn.execute("""
        SELECT t.*,
            s.image_path   AS anchor_image,
            s.slide_number AS anchor_slide_num,
            COUNT(CASE WHEN m.question_type='mcq'          THEN 1 END) AS total_easy,
            COALESCE(SUM(CASE WHEN m.question_type='mcq'          AND m.mastered=1 THEN 1 ELSE 0 END), 0) AS mastered_easy,
            COUNT(CASE WHEN m.question_type='short_answer' THEN 1 END) AS total_medium,
            COALESCE(SUM(CASE WHEN m.question_type='short_answer' AND m.mastered=1 THEN 1 ELSE 0 END), 0) AS mastered_medium,
            COUNT(CASE WHEN m.question_type='applied'      THEN 1 END) AS total_tricky,
            COALESCE(SUM(CASE WHEN m.question_type='applied'      AND m.mastered=1 THEN 1 ELSE 0 END), 0) AS mastered_tricky
        FROM topics t
        LEFT JOIN slides s   ON s.id = t.anchor_slide_id
        LEFT JOIN concepts c ON c.topic_id = t.id
        LEFT JOIN questions q ON q.concept_id = c.id
        LEFT JOIN mastery m   ON m.question_id = q.id
        WHERE t.deck_id = ?
        GROUP BY t.id
        ORDER BY t.order_num
    """, (deck_id,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("anchor_image"):
            d["anchor_image_url"] = f"/slide_images/{Path(d['anchor_image']).name}"
        if d.get("overview_json"):
            try:
                d["overview"] = json.loads(d["overview_json"])
            except Exception:
                d["overview"] = None
        result.append(d)
    return result

@app.get("/topic/{topic_id}")
async def get_topic(topic_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Topic not found")
    d = dict(row)
    if d.get("overview_json"):
        try:
            d["overview"] = json.loads(d["overview_json"])
        except Exception:
            d["overview"] = None
    concepts = conn.execute(
        "SELECT id, concept, explanation, type FROM concepts WHERE topic_id=? ORDER BY id", (topic_id,)
    ).fetchall()
    d["concepts"] = [dict(c) for c in concepts]
    conn.close()
    return d

@app.post("/deck/{deck_id}/reprocess")
async def reprocess_deck(deck_id: int, background_tasks: BackgroundTasks):
    conn = get_db()
    conn.execute("UPDATE decks SET status='processing' WHERE id=?", (deck_id,))
    conn.commit()
    conn.close()
    background_tasks.add_task(reprocess_questions_background, deck_id)
    return {"ok": True}

@app.get("/study/next/{deck_id}")
async def next_question(
    deck_id: int,
    topic_id: int = None,
    difficulty: str = None,
    slide: int = None
):
    q = get_next_question(deck_id, topic_id=topic_id, difficulty=difficulty, target_slide=slide)
    if not q:
        # If topic+difficulty filter found nothing, try without difficulty (maybe they skipped a type)
        if topic_id and difficulty:
            q = get_next_question(deck_id, topic_id=topic_id)
        # If still nothing, try global
        if not q:
            q = get_next_question(deck_id) if (topic_id or difficulty or slide) else None
    if not q:
        return {"done": True}
    return q

@app.get("/weaknesses/{deck_id}")
async def get_weaknesses(deck_id: int):
    conn = get_db()
    rows = conn.execute("""
        SELECT c.concept, c.explanation, m.avg_score, m.attempts, m.question_type,
               s.slide_number, t.name AS topic_name
        FROM mastery m
        JOIN questions q ON q.id = m.question_id
        JOIN concepts c  ON c.id = q.concept_id
        JOIN slides s    ON s.id = q.slide_id
        LEFT JOIN topics t ON t.id = c.topic_id
        WHERE q.deck_id = ? AND m.attempts > 0 AND m.mastered = 0
        ORDER BY m.avg_score ASC, m.attempts DESC
        LIMIT 10
    """, (deck_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

class AnswerBody(BaseModel):
    question_id:   int
    question_type: str
    user_answer:   str
    attempt:       int
    question_data: dict

@app.post("/study/answer")
async def submit_answer(body: AnswerBody):
    if body.question_type == "mcq":
        correct_letter = (body.question_data.get("correct") or "A").strip().upper()[0]
        user_letter    = body.user_answer.strip().upper()[:1]
        is_correct     = user_letter == correct_letter
        score          = 10.0 if is_correct else 0.0
        conn = get_db()
        conn.execute(
            "INSERT INTO responses (question_id, question_type, attempt_number, user_answer, score) VALUES (?,?,?,?,?)",
            (body.question_id, body.question_type, body.attempt, body.user_answer, score)
        )
        conn.commit()
        conn.close()
        update_mastery_db(body.question_id, body.question_type, score)
        return {
            "score":               score,
            "is_correct":          is_correct,
            "correct":             body.question_data.get("correct"),
            "explanation":         body.question_data.get("explanation", ""),
            "follow_up_questions": [],
        }
    else:
        result = await asyncio.to_thread(
            grade_answer_sync,
            body.question_data, body.user_answer, body.attempt, body.question_type
        )
        score = float(result.get("score", 0))
        conn = get_db()
        conn.execute(
            "INSERT INTO responses (question_id, question_type, attempt_number, user_answer, score, feedback_json) VALUES (?,?,?,?,?,?)",
            (body.question_id, body.question_type, body.attempt, body.user_answer, score, json.dumps(result))
        )
        conn.commit()
        conn.close()
        update_mastery_db(body.question_id, body.question_type, score)
        return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
