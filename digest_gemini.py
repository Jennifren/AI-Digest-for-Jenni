#!/usr/bin/env python3
"""
digest_gemini.py — Weekly AI News Digest
-----------------------------------------
Queries the Gemini API (with Google Search grounding) for the week's top AI
news stories plus region-specific AI news, then emails the result as a
styled HTML digest via Gmail SMTP.

Required environment variables:
  GEMINI_API_KEY      — Your Google Gemini API key
  GMAIL_ADDRESS       — Gmail account used to send the email
  GMAIL_APP_PASSWORD  — Gmail App Password (not your regular password)
  RECIPIENT_EMAIL     — Address that receives the digest
  YOUR_CITY_REGION    — (optional) Defaults to "Hampton Roads, Virginia"

Schedule with cron:  0 8 * * 2  /usr/bin/python3 /path/to/digest_gemini.py
"""

import os
import smtplib
import sys
import textwrap
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import google.generativeai as genai
from google.generativeai.types import Tool, GoogleSearchRetrieval  # grounding


# ---------------------------------------------------------------------------
# 1. Configuration — all values from environment variables
# ---------------------------------------------------------------------------

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY")
GMAIL_ADDRESS    = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PW     = os.environ.get("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL  = os.environ.get("RECIPIENT_EMAIL")
CITY_REGION      = os.environ.get("YOUR_CITY_REGION", "Hampton Roads, Virginia")

_MISSING = [
    name for name, val in {
        "GEMINI_API_KEY":     GEMINI_API_KEY,
        "GMAIL_ADDRESS":      GMAIL_ADDRESS,
        "GMAIL_APP_PASSWORD": GMAIL_APP_PW,
        "RECIPIENT_EMAIL":    RECIPIENT_EMAIL,
    }.items() if not val
]
if _MISSING:
    print(f"[ERROR] Missing environment variable(s): {', '.join(_MISSING)}", file=sys.stderr)
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME    = "gemini-1.5-pro"          # Switch to "gemini-1.5-flash" for lower cost
TODAY         = datetime.now().strftime("%B %d, %Y")
WEEK_AGO      = (datetime.now() - timedelta(days=7)).strftime("%B %d, %Y")


# ---------------------------------------------------------------------------
# 2. Gemini helpers
# ---------------------------------------------------------------------------

def _build_model_with_search() -> genai.GenerativeModel:
    """Return a GenerativeModel with Google Search grounding enabled."""
    search_tool = Tool(google_search_retrieval=GoogleSearchRetrieval())
    return genai.GenerativeModel(
        model_name=MODEL_NAME,
        tools=[search_tool],
    )


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini (with Search grounding) and return the text."""
    model = _build_model_with_search()
    response = model.generate_content(prompt)
    # Flatten all text parts in case the response is multi-part
    parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# 3. Prompt builders
# ---------------------------------------------------------------------------

GLOBAL_PROMPT = textwrap.dedent(f"""
    Today is {TODAY}.  Search the web for real AI news published between
    {WEEK_AGO} and {TODAY}.

    Return EXACTLY 5 of the most important global AI news stories from the
    past 7 days.  For each story produce a block in this exact format
    (use the literal delimiter lines shown):

    ---STORY---
    HEADLINE: <concise headline>
    SUMMARY: <2-3 sentence plain-English summary>
    WHY_IT_MATTERS: <1-2 sentences on broader significance>
    SOURCE_LABEL: <publication or website name>
    SOURCE_URL: <direct URL to the article>
    ---END---

    Rules:
    - Every field must be present and on its own line.
    - Do NOT use markdown bold, bullets, or headers inside the fields.
    - SOURCE_URL must be a real, reachable URL found via search.
    - Order stories by newsworthiness (most important first).
""").strip()


REGIONAL_PROMPT = textwrap.dedent(f"""
    Today is {TODAY}.  Search the web for AI-related news specific to
    {CITY_REGION} published between {WEEK_AGO} and {TODAY}.

    Include local universities, companies, startups, government / policy
    actions, and community events related to AI.

    Return UP TO 4 stories (fewer is fine if genuine local results are
    scarce — never fabricate).  Use the same exact format:

    ---STORY---
    HEADLINE: <concise headline>
    SUMMARY: <2-3 sentence plain-English summary>
    WHY_IT_MATTERS: <1-2 sentences on local significance>
    SOURCE_LABEL: <publication or website name>
    SOURCE_URL: <direct URL to the article>
    ---END---

    If no relevant local AI news exists this week, return exactly:
    NO_LOCAL_NEWS
""").strip()


# ---------------------------------------------------------------------------
# 4. Response parser
# ---------------------------------------------------------------------------

def parse_stories(raw: str) -> list[dict]:
    """Parse the delimited story blocks returned by Gemini."""
    stories = []
    blocks = raw.split("---STORY---")
    for block in blocks:
        end_idx = block.find("---END---")
        if end_idx == -1:
            continue
        block = block[:end_idx].strip()
        story: dict = {}
        for line in block.splitlines():
            for key in ("HEADLINE", "SUMMARY", "WHY_IT_MATTERS",
                        "SOURCE_LABEL", "SOURCE_URL"):
                if line.startswith(f"{key}:"):
                    story[key] = line[len(key) + 1:].strip()
                    break
        if story.get("HEADLINE"):          # require at minimum a headline
            stories.append(story)
    return stories


# ---------------------------------------------------------------------------
# 5. HTML builder
# ---------------------------------------------------------------------------

STYLES = """
    body { margin:0; padding:0; background:#f4f4f7; font-family:
           'Helvetica Neue', Helvetica, Arial, sans-serif; color:#2d2d2d; }
    .wrapper { max-width:680px; margin:32px auto; background:#ffffff;
               border-radius:10px; overflow:hidden;
               box-shadow:0 2px 12px rgba(0,0,0,.10); }
    .header { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
              padding:36px 40px; }
    .header h1 { margin:0; color:#ffffff; font-size:26px; letter-spacing:-.3px; }
    .header p  { margin:6px 0 0; color:#94b4d0; font-size:14px; }
    .section-label { background:#f0f4ff; border-left:4px solid #3b82f6;
                     margin:0; padding:14px 40px; font-size:13px;
                     font-weight:700; letter-spacing:.8px;
                     text-transform:uppercase; color:#3b5998; }
    .story { padding:26px 40px; border-bottom:1px solid #eef0f4; }
    .story:last-child { border-bottom:none; }
    .story h2 { margin:0 0 10px; font-size:18px; line-height:1.35;
                color:#0f172a; }
    .story p  { margin:0 0 8px; font-size:14.5px; line-height:1.6;
                color:#4b5563; }
    .label    { font-size:12px; font-weight:700; text-transform:uppercase;
                letter-spacing:.5px; color:#6b7280; margin:0 0 4px; }
    .why      { background:#f8faff; border-radius:6px; padding:10px 14px;
                font-size:14px; color:#374151; margin:10px 0; }
    .source-link a { font-size:13px; color:#2563eb; text-decoration:none; }
    .source-link a:hover { text-decoration:underline; }
    .no-local { padding:26px 40px; font-size:14.5px; color:#6b7280;
                font-style:italic; }
    .footer   { background:#f9fafb; padding:20px 40px;
                border-top:1px solid #e5e7eb; font-size:12px;
                color:#9ca3af; text-align:center; }
"""


def story_html(story: dict, index: int) -> str:
    headline    = story.get("HEADLINE",       "Untitled")
    summary     = story.get("SUMMARY",        "")
    why         = story.get("WHY_IT_MATTERS", "")
    src_label   = story.get("SOURCE_LABEL",   "Source")
    src_url     = story.get("SOURCE_URL",     "#")

    # Basic URL sanitisation — prevents obviously broken hrefs
    if not src_url.startswith(("http://", "https://")):
        src_url = "#"

    return f"""
    <div class="story">
      <h2>{index}. {headline}</h2>
      <p class="label">Summary</p>
      <p>{summary}</p>
      {'<p class="label">Why it matters</p><div class="why">' + why + '</div>' if why else ''}
      <p class="source-link">
        <a href="{src_url}" target="_blank" rel="noopener noreferrer">
          ↗ {src_label}
        </a>
      </p>
    </div>
    """


def build_html(global_stories: list[dict],
               local_stories: list[dict],
               no_local: bool) -> str:

    global_html = "".join(
        story_html(s, i + 1) for i, s in enumerate(global_stories)
    )

    if no_local:
        local_content = (
            '<p class="no-local">No AI-specific local news found for '
            f'{CITY_REGION} this week. Check back next Tuesday!</p>'
        )
    else:
        local_content = "".join(
            story_html(s, i + 1) for i, s in enumerate(local_stories)
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Weekly AI Digest — {TODAY}</title>
  <style>{STYLES}</style>
</head>
<body>
<div class="wrapper">

  <!-- Header -->
  <div class="header">
    <h1>🤖 Weekly AI Digest</h1>
    <p>Top stories for the week of {TODAY}</p>
  </div>

  <!-- Section 1: Global -->
  <p class="section-label">🌐 &nbsp;Top 5 Global AI Stories</p>
  {global_html}

  <!-- Section 2: Regional -->
  <p class="section-label">📍 &nbsp;AI News — {CITY_REGION}</p>
  {local_content}

  <!-- Footer -->
  <div class="footer">
    Generated by Gemini {MODEL_NAME} with Google Search grounding.<br>
    Delivered every Tuesday morning. Unsubscribe by removing the cron job.
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 6. Email sender
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str) -> None:
    """Send the HTML digest via Gmail SMTP (port 587, STARTTLS)."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = RECIPIENT_EMAIL

    # Attach plain-text fallback then HTML
    plain = "Your email client does not support HTML. Please view in a modern client."
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    print(f"[INFO] Connecting to smtp.gmail.com:587 …")
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PW)
        server.sendmail(GMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
    print(f"[INFO] Email sent to {RECIPIENT_EMAIL}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Generating digest for {TODAY} …")

    # --- Global news ---
    print("[INFO] Querying Gemini for global AI news …")
    global_raw = ask_gemini(GLOBAL_PROMPT)
    global_stories = parse_stories(global_raw)
    print(f"[INFO] Parsed {len(global_stories)} global stories.")

    if not global_stories:
        print("[WARN] No global stories parsed. Check Gemini output:", file=sys.stderr)
        print(global_raw[:800], file=sys.stderr)

    # --- Regional news ---
    print(f"[INFO] Querying Gemini for AI news in {CITY_REGION} …")
    local_raw = ask_gemini(REGIONAL_PROMPT)
    no_local = "NO_LOCAL_NEWS" in local_raw and "---STORY---" not in local_raw
    local_stories = [] if no_local else parse_stories(local_raw)
    print(f"[INFO] Parsed {len(local_stories)} local stories (no_local={no_local}).")

    # --- Build & send ---
    html = build_html(global_stories, local_stories, no_local)
    subject = f"🤖 Weekly AI Digest — {TODAY}"
    send_email(subject, html)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
