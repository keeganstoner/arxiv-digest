import json
import os
import re
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
import feedparser
import requests

# ── Configuration ────────────────────────────────────────────────────────────

_config_path = os.environ.get("CONFIG_FILE", "configs/security.json")
with open(_config_path) as _f:
    _config = json.load(_f)

DIGEST_NAME   = _config["name"]
RSS_FEEDS     = _config["feeds"]
KEYWORDS      = _config["keywords"]
TOP_N         = _config.get("top_n", 5)

LOOKBACK_HOURS = 36     # safety net for weekend gaps in ArXiv announcements

# Pricing per million tokens (verify at console.anthropic.com/settings/billing)
SONNET_PRICE  = {"input": 3.00,  "output": 15.00}
HAIKU_PRICE   = {"input": 0.80,  "output": 4.00}

SUMMARY_PROMPT = """You are explaining an academic paper to a security researcher who is smart and technical but may not be deep in this specific subfield.

Title: {title}
Authors: {authors}
Published: {published}
ArXiv ID: {arxiv_id}

Abstract:
{abstract}

Write four sections with these headers (use **Header** markdown):

**Background**: 2-3 sentences of context a non-specialist would need. What is the broader problem space? What are the key techniques or concepts this paper builds on? Write this like the first paragraph of a good blog post — assume the reader is technical but hasn't read papers in this area.

**What they did**: 3-5 sentences. Explain the actual contribution directly and concretely, as if telling a colleague over coffee. Don't use phrases like "the authors propose" or "this paper presents" — just say what it is and what it does. If it's a tool or system, say what the tool does. If it's an attack, say how the attack works.

**Why it matters**: 2-3 sentences on the real-world significance. Be specific — who is affected, what changes if this work is adopted or if attackers use it?

**Limitations**: 1-2 sentences on what's missing or what the paper doesn't address."""

# ── ArXiv fetching ────────────────────────────────────────────────────────────

def fetch_recent_papers(lookback_hours: int) -> list[dict]:
    headers = {"User-Agent": "arxiv-digest/1.0 (keeganstoner@gmail.com)"}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    seen_ids = set()
    papers = []

    for feed_url in RSS_FEEDS:
        print(f"  Fetching {feed_url}...")
        response = requests.get(feed_url, timeout=30, headers=headers)
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        for entry in feed.entries:
            arxiv_id = entry.link.split("/abs/")[-1]
            if arxiv_id in seen_ids:
                continue

            title = entry.title.replace("\n", " ").strip()
            abstract = re.sub(r"<[^>]+>", "", entry.summary).replace("\n", " ").strip()

            if not any(kw in (title + " " + abstract).lower() for kw in KEYWORDS):
                continue

            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if published < cutoff:
                    continue
                published_str = published.strftime("%Y-%m-%d %H:%M UTC")
            else:
                published_str = "Unknown"

            if hasattr(entry, "authors"):
                authors = ", ".join(a.name for a in entry.authors)
            else:
                authors = getattr(entry, "author", "Unknown")

            seen_ids.add(arxiv_id)
            papers.append({
                "title": title,
                "authors": authors,
                "published": published_str,
                "abstract": abstract,
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

def select_top_papers(client: anthropic.Anthropic, papers: list[dict], top_n: int) -> tuple[list[dict], dict]:
    if len(papers) <= top_n:
        return papers, {"input": 0, "output": 0}

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
    usage = {"input": message.usage.input_tokens, "output": message.usage.output_tokens}
    raw = message.content[0].text.strip()
    match = re.search(r"\[[\d,\s]+\]", raw)
    if not match:
        print(f"  Warning: could not parse ranking response '{raw}', using first {top_n}")
        return papers[:top_n], usage
    indices = json.loads(match.group())
    valid = [i for i in indices if 0 <= i < len(papers)][:top_n]
    return [papers[i] for i in valid], usage

# ── Summarization ─────────────────────────────────────────────────────────────

def summarize_paper(client: anthropic.Anthropic, paper: dict) -> tuple[str, dict]:
    prompt = SUMMARY_PROMPT.format(**paper)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = {"input": message.usage.input_tokens, "output": message.usage.output_tokens}
    return message.content[0].text.strip(), usage

# ── Email formatting ──────────────────────────────────────────────────────────

def build_email_html(papers_with_summaries: list[tuple[dict, str]], all_papers: list[dict], date_str: str, cost_str: str = "") -> str:
    summarized_ids = {p["arxiv_id"] for p, _ in papers_with_summaries}

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
          <p style="margin:0 0 12px; font-size:14px; line-height:1.6;">{html_summary}</p>
          <details style="margin-top:8px;">
            <summary style="cursor:pointer; color:#555; font-size:12px;">Abstract</summary>
            <p style="margin:8px 0 0; font-size:13px; line-height:1.5; color:#444;">{paper['abstract']}</p>
          </details>
        </div>
        """)

    body = "\n".join(sections) if sections else "<p>No new papers matched today.</p>"

    remaining = [p for p in all_papers if p["arxiv_id"] not in summarized_ids]
    also_section = ""
    if remaining:
        items = "".join(
            f'<li style="margin-bottom:6px;"><a href="{p["abs_url"]}">{p["title"]}</a>'
            f' <a href="{p["pdf_url"]}" style="color:#888; font-size:12px;">[pdf]</a></li>'
            for p in remaining
        )
        also_section = f"""
        <h2 style="font-size:16px; margin-top:40px; border-top:1px solid #ddd; padding-top:16px;">
          Also matched ({len(remaining)})
        </h2>
        <ul style="padding-left:20px; font-size:14px; line-height:1.6;">{items}</ul>
        """

    return f"""
    <html><body style="font-family:Georgia,serif; max-width:700px; margin:auto; padding:24px; color:#222;">
      <h1 style="font-size:20px; border-bottom:1px solid #ddd; padding-bottom:8px;">
        ArXiv Digest: {DIGEST_NAME} — {date_str}
      </h1>
      <p style="color:#555; font-size:13px;">{len(papers_with_summaries)} summaries · {len(all_papers)} total matches</p>
      {body}
      {also_section}
      <hr style="margin-top:40px;">
      <p style="color:#888; font-size:12px;">Generated by arxiv-digest ·
        <a href="https://github.com/keeganstoner/arxiv-digest">source</a>
        {f"· {cost_str}" if cost_str else ""}
      </p>
    </body></html>
    """

def build_email_plaintext(papers_with_summaries: list[tuple[dict, str]], all_papers: list[dict], date_str: str, cost_str: str = "") -> str:
    summarized_ids = {p["arxiv_id"] for p, _ in papers_with_summaries}
    lines = [f"ArXiv Digest: {DIGEST_NAME} — {date_str}", f"{len(papers_with_summaries)} summaries · {len(all_papers)} total matches", "=" * 60]
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
            "",
            "Abstract:",
            paper["abstract"],
            "-" * 60,
        ]
    if not papers_with_summaries:
        lines.append("\nNo new papers matched today.")
    remaining = [p for p in all_papers if p["arxiv_id"] not in summarized_ids]
    if remaining:
        lines += ["", f"Also matched ({len(remaining)}):", ""]
        for p in remaining:
            lines += [f"  {p['title']}", f"  {p['abs_url']}", ""]
    if cost_str:
        lines += ["", cost_str]
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

    papers = fetch_recent_papers(LOOKBACK_HOURS)[:50]
    print(f"Found {len(papers)} matching papers in the last {LOOKBACK_HOURS}h")

    if not papers:
        print("Nothing to summarize; skipping email.")
        return

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    selected, rank_usage = select_top_papers(client, papers, TOP_N)
    print(f"Summarizing {len(selected)} papers...")

    papers_with_summaries = []
    sonnet_usage = {"input": 0, "output": 0}
    for i, paper in enumerate(selected, 1):
        print(f"  [{i}/{len(selected)}] {paper['title'][:70]}...")
        summary, usage = summarize_paper(client, paper)
        sonnet_usage["input"] += usage["input"]
        sonnet_usage["output"] += usage["output"]
        papers_with_summaries.append((paper, summary))

    haiku_cost = (rank_usage["input"] * HAIKU_PRICE["input"] + rank_usage["output"] * HAIKU_PRICE["output"]) / 1_000_000
    sonnet_cost = (sonnet_usage["input"] * SONNET_PRICE["input"] + sonnet_usage["output"] * SONNET_PRICE["output"]) / 1_000_000
    total_cost = haiku_cost + sonnet_cost
    cost_str = f"API cost: ${total_cost:.4f}"
    print(cost_str)

    html = build_email_html(papers_with_summaries, papers, date_str, cost_str)
    plain = build_email_plaintext(papers_with_summaries, papers, date_str, cost_str)
    subject = f"ArXiv Digest: {DIGEST_NAME} {date_str} ({len(selected)} papers)"

    print("Sending email...")
    send_email(subject, html, plain)
    print("Done.")

if __name__ == "__main__":
    main()
