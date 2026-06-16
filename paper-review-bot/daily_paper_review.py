#!/usr/bin/env python3
"""
Daily BraTS-MET paper review bot.

Searches the web for new papers relevant to BraTS 2026 Challenge 1 (brain
metastases segmentation, pre/post-treatment), reviews the core technique and
architecture of each against the project's nnU-Net baseline, appends a dated
entry to PAPER_REVIEWS.md, and emails the result via Gmail SMTP.

Runs entirely on GitHub Actions -- no dependency on any local machine.

Required environment variables (set as GitHub Actions repo secrets):
  ANTHROPIC_API_KEY   - Anthropic API key (console.anthropic.com)
  GMAIL_ADDRESS       - Gmail address used to send the email
  GMAIL_APP_PASSWORD  - 16-character Gmail App Password (NOT your normal password)
  RECIPIENT_EMAIL     - Where to send the daily review (e.g. jangirvineet2@gmail.com)
"""

import os
import re
import sys
import smtplib
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import anthropic

LOG_PATH = "PAPER_REVIEWS.md"
LOG_HEADER = "# BraTS 2026 MET — Daily Paper Review Log\n"

COMPETITION_CONTEXT = """
COMPETITION CONTEXT:
- Task: multi-class 3D segmentation of brain metastases lesions on multi-modal MRI
  (T1c, T1n, T2f, T2w). Labels: 1=NETC (non-enhancing tumor core), 2=SNFH (surrounding
  non-enhancing FLAIR hyperintensity/edema), 3=ET (enhancing tumor), 4=RC (resection
  cavity, post-treatment/UCSD cases only).
- Dataset: ~1296 training timepoints, 75% multi-focal lesions, many lesions are small
  (sub-detection threshold ~27mm^3), mix of pre-treatment (no RC) and post-treatment
  (with RC) cases, two coordinate spaces (SRI24 atlas vs native UCSD clinical space).
- Current baseline: nnU-Net v2, 3d_fullres config, 6-stage U-Net (features
  32->64->128->256->320->320), InstanceNorm3d, LeakyReLU, Dice+CE loss, patch
  128x160x112. Results (fold 0, 20 held-out cases): Dice NETC=0.460, SNFH=0.596,
  ET=0.661. NETC is the weakest class -- small/non-enhancing structures are hardest.
- Goal: surface papers whose techniques could plausibly improve on this nnU-Net
  baseline, especially for small-lesion sensitivity, multi-focal lesion detection,
  NETC/RC boundary precision, or multi-modal fusion -- and flag which are realistic
  to implement on a single-GPU (RTX 5090 class) setup.
"""


def read_existing_titles(path: str) -> list:
    if not os.path.exists(path):
        return []
    text = open(path, encoding="utf-8").read()
    # Titles are written as "### N. [Title](link)"
    return re.findall(r"^### \d+\.\s*\[(.+?)\]\(", text, flags=re.MULTILINE)


def build_prompt(already_covered: list) -> str:
    covered_block = (
        "\n".join(f"- {t}" for t in already_covered) if already_covered else "(none yet)"
    )
    return f"""You are reviewing research papers for a BraTS 2026 Challenge 1 competition entry
(brain metastases MRI segmentation, pre- and post-treatment).

{COMPETITION_CONTEXT}

Papers already reviewed in previous runs (DO NOT repeat these):
{covered_block}

TASK:
Use web search to find up to 10 papers NOT in the list above, prioritizing:
1. Brain metastases segmentation / BraTS-MET specifically.
2. Recent general brain tumor / lesion segmentation architecture papers (nnU-Net
   extensions, transformer/Mamba-based 3D medical segmentation, small-object
   segmentation, multi-modal MRI fusion) applicable even if not metastases-specific.
3. Prefer arXiv preprints / MICCAI/BraTS workshop papers from the last ~30 days; if
   fewer than 10 genuinely new papers exist, fill remaining slots with relevant
   papers from the last 6 months not yet covered, and say so.
If fewer than 10 relevant unread papers exist at all, return as many as you found.

For each paper write:
- Title, authors (first author + "et al." if many), venue/date, link.
- Core technique/architecture in 2-4 sentences.
- Reported results if stated, noting if not directly comparable to BraTS-MET.
- Verdict: one of "Strong candidate — implement", "Worth testing/ablating",
  "Interesting but low priority", or "Not relevant/recommended", with 1-2 sentences
  of reasoning tied to the competition context (e.g. weak NETC Dice, small lesions,
  multi-focal detection).

OUTPUT FORMAT — respond with ONLY this markdown, nothing else before or after:

[1-2 sentence summary of today's haul and the single best pick]

### 1. [Title](link)
- **Technique/architecture:** ...
- **Results:** ...
- **Verdict:** ...

(repeat, numbered, for every paper found)
"""


def run_review() -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    covered = read_existing_titles(LOG_PATH)
    prompt = build_prompt(covered)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 20}],
        messages=[{"role": "user", "content": prompt}],
    )

    parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(parts).strip()


def append_log(content: str) -> None:
    today = date.today().isoformat()
    entry = f"\n## {today}\n\n{content}\n\n---\n"
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(LOG_HEADER)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


def send_email(content: str) -> None:
    import markdown as md_lib

    today = date.today().isoformat()

    sender = os.environ["GMAIL_ADDRESS"]
    password = os.environ["GMAIL_APP_PASSWORD"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    html_body = md_lib.markdown(content, extensions=["extra"])

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"BraTS-MET Daily Paper Review — {today}"
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(content, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, [recipient], msg.as_string())


def main() -> None:
    content = run_review()
    if not content:
        print("No content returned from model; skipping log/email update.")
        sys.exit(1)
    append_log(content)
    send_email(content)
    print("Done. Log updated and email sent.")


if __name__ == "__main__":
    main()
