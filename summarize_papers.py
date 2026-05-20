import json
import os
import re
import smtplib
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
import feedparser

# ── Configuration ────────────────────────────────────────────────────────────

# ArXiv search query. Uses the ArXiv query syntax:
#   all:TERM     searches all fields (title, abstract, authors)
#   ti:TERM      title only
#   abs:TERM     abstract only
#   cat:CATEGORY category (e.g. cs.AI, cs.LG, q-bio.NC)
#   AND / OR / ANDNOT for boolean logic
ARXIV_QUERY = (                                           
    "(cat:cs.CR OR cat:cs.NI) AND ("
    'abs:"LTE security" OR abs:"5G security" OR abs:"5G vulnerability" OR '                                                  
    'abs:"baseband vulnerability" OR abs:"baseband exploit" OR '                                                             
    'abs:"IMSI catcher" OR abs:"IMSI-catcher" OR abs:"false base station" OR '                                               
    'abs:"fake base station" OR abs:"rogue base station" OR abs:"cell site simulator" OR '                                   
    'abs:"traffic fingerprinting" OR abs:"website fingerprinting" OR '                                                       
    'abs:"censorship circumvention" OR abs:"censorship evasion" OR '                                                         
    'abs:"great firewall" OR abs:"domain fronting" OR abs:"pluggable transport" OR '                                         
    'abs:"protocol obfuscation"'                                                                                             
    ")"                                                                                                                      
)                                                                                                                            
     

MAX_RESULTS = 50        # papers to fetch per run (ArXiv returns newest first)
LOOKBACK_HOURS = 26     # include papers published within this window
TOP_N = 5               # max papers to summarize and email

SUMMARY_PROMPT = """You are summarizing an academic paper for a researcher who wants a quick but substantive overview.

Title: {title}
Authors: {authors}
Published: {published}
ArXiv ID: {arxiv_id}

Abstract:
{abstract}

Write a summary with these sections:
**What they did** (1-2 sentences): The core contribution or finding.
**Why it matters** (1 sentence): The significance or potential impact.
**Key idea** (2-3 sentences): The technical approach or main insight.
**Limitations / caveats** (1 sentence): What the authors acknowledge as limitations, or what seems missing.

Be direct and concrete. Avoid filler phrases like "the authors propose" — just state what was done."""

# ── ArXiv fetching ────────────────────────────────────────────────────────────

def fetch_recent_papers(query: str, max_results: int, lookback_hours: int) -> list[dict]:
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query={quote(query, safe=':\"()')}"
        f"&sortBy=submittedDate&sortOrder=descending"
        f"&max_results={max_results}"
    )
    feed = feedparser.parse(url)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    papers = []
    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue
        arxiv_id = entry.id.split("/abs/")[-1]
        papers.append({
            "title": entry.title.replace("\n", " ").strip(),
            "authors": ", ".join(a.name for a in entry.authors),
            "published": published.strftime("%Y-%m-%d %H:%M UTC"),
            "abstract": entry.summary.replace("\n", " ").strip(),
            "arxiv_id": arxiv_id,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        })

    return papers

# ── Ranking ───────────────────────────────────────────────────────────────────

RANKING_PROMPT = """You are helping filter academic papers for a security researcher. Their core interests are:
1. LTE/5G protocol vulnerabilities and cellular network security
2. Fake base stations, IMSI catchers, and cell site simulator detection/defense
3. IP-level traffic fingerprinting and obfuscation techniques
4. Censorship evasion and circumvention tools and protocols

Below are {n} papers that passed a keyword filter. Select the {top_n} most directly relevant to these interests. Prefer papers that are central to one of these topics over papers that merely mention them in passing.

Reply with ONLY a JSON array of the selected paper indices (0-based), ordered from most to least relevant. Example: [2, 0, 4, 1, 3]

Papers:
{paper_list}"""

def select_top_papers(client: anthropic.Anthropic, papers: list[dict], top_n: int) -> list[dict]:
    if len(papers) <= top_n:
        return papers

    print(f"  Ranking {len(papers)} papers to select top {top_n}...")
    paper_list = "\n\n".join(
        f"[{i}] {p['title']}\n{p['abstract'][:400]}"
        for i, p in enumerate(papers)
    )
    prompt = RANKING_PROMPT.format(n=len(papers), top_n=top_n, paper_list=paper_list)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    match = re.search(r"\[[\d,\s]+\]", raw)
    if not match:
        print(f"  Warning: could not parse ranking response '{raw}', using first {top_n}")
        return papers[:top_n]
    indices = json.loads(match.group())
    valid = [i for i in indices if 0 <= i < len(papers)][:top_n]
    return [papers[i] for i in valid]

# ── Summarization ─────────────────────────────────────────────────────────────

def summarize_paper(client: anthropic.Anthropic, paper: dict) -> str:
    prompt = SUMMARY_PROMPT.format(**paper)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()

# ── Email formatting ──────────────────────────────────────────────────────────

def build_email_html(papers_with_summaries: list[tuple[dict, str]], date_str: str) -> str:
    sections = []
    for paper, summary in papers_with_summaries:
        html_summary = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", summary)
        html_summary = html_summary.replace("\n", "<br>")

        sections.append(f"""
        <div style="margin-bottom:36px; border-left:3px solid #4a90d9; padding-left:16px;">
          <h2 style="margin:0 0 4px; font-size:16px;">
            <a href="{paper['abs_url']}" style="color:#1a0dab; text-decoration:none;">{paper['title']}</a>
          </h2>
          <p style="margin:0 0 8px; color:#555; font-size:13px;">
            {paper['authors']} · {paper['published']}
            &nbsp;|&nbsp;
            <a href="{paper['abs_url']}">Abstract</a>
            &nbsp;·&nbsp;
            <a href="{paper['pdf_url']}">PDF</a>
          </p>
          <p style="margin:0; font-size:14px; line-height:1.6;">{html_summary}</p>
        </div>
        """)

    body = "\n".join(sections) if sections else "<p>No new papers matched today.</p>"

    return f"""
    <html><body style="font-family:Georgia,serif; max-width:700px; margin:auto; padding:24px; color:#222;">
      <h1 style="font-size:20px; border-bottom:1px solid #ddd; padding-bottom:8px;">
        ArXiv Digest — {date_str}
      </h1>
      <p style="color:#555; font-size:13px;">{len(papers_with_summaries)} papers</p>
      {body}
      <hr style="margin-top:40px;">
      <p style="color:#888; font-size:12px;">Generated by arxiv-digest ·
        <a href="https://github.com/keeganstoner/arxiv-digest">source</a>
      </p>
    </body></html>
    """

def build_email_plaintext(papers_with_summaries: list[tuple[dict, str]], date_str: str) -> str:
    lines = [f"ArXiv Digest — {date_str}", f"{len(papers_with_summaries)} papers", "=" * 60]
    for paper, summary in papers_with_summaries:
        lines += [
            "",
            paper["title"],
            paper["authors"],
            paper["published"],
            f"Abstract: {paper['abs_url']}",
            f"PDF:      {paper['pdf_url']}",
            "",
            summary,
            "-" * 60,
        ]
    if not papers_with_summaries:
        lines.append("\nNo new papers matched today.")
    return "\n".join(lines)

# ── Email sending ─────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str, plain_body: str) -> None:
    sender = os.environ["EMAIL_ADDRESS"]
    password = os.environ["EMAIL_APP_PASSWORD"]
    recipients = [r.strip() for r in os.environ["RECIPIENT_EMAILS"].split(",")]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Fetching papers for {date_str}...")

    papers = fetch_recent_papers(ARXIV_QUERY, MAX_RESULTS, LOOKBACK_HOURS)
    print(f"Found {len(papers)} papers in the last {LOOKBACK_HOURS}h")

    if not papers:
        print("Nothing to summarize; skipping email.")
        return

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    selected = select_top_papers(client, papers, TOP_N)
    print(f"Summarizing {len(selected)} papers...")

    papers_with_summaries = []
    for i, paper in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {paper['title'][:70]}...")
        summary = summarize_paper(client, paper)
        papers_with_summaries.append((paper, summary))

    html = build_email_html(papers_with_summaries, date_str)
    plain = build_email_plaintext(papers_with_summaries, date_str)
    subject = f"ArXiv Digest {date_str} ({len(selected)} papers)"

    print("Sending email...")
    send_email(subject, html, plain)
    print("Done.")

if __name__ == "__main__":
    main()
