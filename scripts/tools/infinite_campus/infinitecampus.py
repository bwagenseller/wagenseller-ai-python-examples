"""
infinitecampus.py
------------
Fetches your trimester grades AND recent assignments from Infinite Campus (Boyertown)
using Microsoft SSO login via Playwright, then hits the IC JSON APIs.
Prints results to the terminal and emails a summary via Gmail.

SETUP (run these in your terminal first):
  python -m pip install playwright python-dotenv
  python -m playwright install chromium

  The chromium step is required and separate from pip: playwright ships no browser
  in its wheel, and without it launch() fails with "Executable doesn't exist".
  If chromium then fails to start on a missing shared library, install its system
  deps with: sudo python -m playwright install-deps chromium

  This script also needs SECRETS_FILE set in the SHELL environment (not in the
  .env itself) - it is read via os.getenv BEFORE load_dotenv() runs, and points at
  the .env holding the INFINITE_CAMPUS_* credentials. See .env.example.

HOW TO RUN:
  python infinitecampus.py

Examples:

python infinitecampus.py --print    # human-readable terminal output
python infinitecampus.py --email    # sends email silently
python infinitecampus.py --json     # JSON to stdout + saves ic_output.json

python infinitecampus.py --print                        # T2, last 14 days (defaults)
python infinitecampus.py --print --term T1              # T1 grades
python infinitecampus.py --print --days 7               # only last 7 days of assignments
python infinitecampus.py --email --term T2 --days 30    # email with 30 days of assignments
python infinitecampus.py --json --term T3               # JSON for T3
python infinitecampus.py --help                         # see all options
"""

from playwright.sync_api import sync_playwright
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import time
from dotenv import load_dotenv
import os

# ─────────────────────────────────────────────
# 🔧 CONFIGURATION — fill these in!
# ─────────────────────────────────────────────

SECRETS_FILE=os.getenv("SECRETS_FILE")
load_dotenv(SECRETS_FILE)

INFINITE_CAMPUS_MS_SSO_LOGIN = os.getenv("INFINITE_CAMPUS_MS_SSO_LOGIN")
INFINITE_CAMPUS_MS_SSO_PASSWORD = os.getenv("INFINITE_CAMPUS_MS_SSO_PASSWORD")
INFINITE_CAMPUS_FROM_EMAIL = os.getenv("INFINITE_CAMPUS_FROM_EMAIL")
INFINITE_CAMPUS_FROM_PASSWORD = os.getenv("INFINITE_CAMPUS_FROM_PASSWORD")
INFINITE_CAMPUS_EMAIL_RECIPIENTS = os.getenv("INFINITE_CAMPUS_EMAIL_RECIPIENTS")
INFINITE_CAMPUS_LOGIN_URL = os.getenv("INFINITE_CAMPUS_LOGIN_URL")
INFINITE_CAMPUS_GRADES_API = os.getenv("INFINITE_CAMPUS_GRADES_API")
INFINITE_CAMPUS_ASSIGN_API = os.getenv("INFINITE_CAMPUS_ASSIGN_API")

TARGET_TERM     = "T3"   # Change to "T1" or "T3" if needed
ASSIGNMENT_DAYS = 14     # How many days back to show assignments

# Set to False to watch the browser (good for debugging), True to run silently
HEADLESS = False

# ─────────────────────────────────────────────


def fetch_json(page, url):
    """Navigate to a JSON API URL and return parsed JSON."""
    page.goto(url, wait_until="networkidle", timeout=15000)
    raw = page.content()
    # Browser wraps JSON in <html><body><pre>...</pre></body></html>
    start = raw.find("[")
    end_bracket = raw.rfind("]") + 1
    start_brace = raw.find("{")
    end_brace = raw.rfind("}") + 1

    # Figure out whether it's an array or object
    if start != -1 and (start_brace == -1 or start < start_brace):
        return json.loads(raw[start:end_bracket])
    elif start_brace != -1:
        return json.loads(raw[start_brace:end_brace])
    else:
        return None


def login_and_fetch(max_retries=3):
    """Launch browser, log in via Microsoft SSO, fetch grades + assignments.
    Retries up to max_retries times if login gets stuck."""

    for attempt in range(1, max_retries + 1):
        print(f"Login attempt {attempt} of {max_retries}...")
        try:
            result = _try_login_and_fetch()
            if result != (None, None):
                return result
            print("Login returned no data, retrying...")
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print("Retrying in 3 seconds...")
                time.sleep(3)

    print("All login attempts failed.")
    return None, None


def _try_login_and_fetch():
    """Single login attempt — called by login_and_fetch."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        page = context.new_page()

        # Step 1: Go to IC login page and click SSO button
        print("  Opening Infinite Campus login page...")
        page.goto(INFINITE_CAMPUS_LOGIN_URL, wait_until="domcontentloaded")
        page.wait_for_selector("#samlloginlink", timeout=8000)
        page.click("#samlloginlink")
        print("  Clicked SSO button.")

        # Step 2: Fill in Microsoft credentials
        print("  Entering Microsoft credentials...")
        page.wait_for_selector("input[type='email']", timeout=10000)
        page.fill("input[type='email']", INFINITE_CAMPUS_MS_SSO_LOGIN)
        page.keyboard.press("Enter")
        page.wait_for_selector("input[type='password']", timeout=10000)
        time.sleep(1)  # let the password field fully render
        page.fill("input[type='password']", INFINITE_CAMPUS_MS_SSO_PASSWORD)
        page.keyboard.press("Enter")

        # Step 3: Handle "Stay signed in?" prompt — wait for it and always click No
        try:
            page.wait_for_selector("input[id='idBtn_Back'], input[value='No'], button:has-text('No')", timeout=8000)
            page.locator("input[id='idBtn_Back'], input[value='No'], button:has-text('No')").first.click()
            print("  Dismissed 'Stay signed in?' prompt.")
        except Exception:
            print("  No 'Stay signed in?' prompt appeared, continuing...")

            pass

        # Step 4: Wait for IC to load after SSO redirect
        print("  Waiting for Infinite Campus to load...")
        try:
            page.wait_for_url("**/campus/**", timeout=15000)
        except Exception:
            print("  Timed out waiting for redirect — trying anyway...")
        page.wait_for_load_state("networkidle", timeout=10000)

        # Step 5: Fetch grades JSON
        print("  Fetching grades from IC API...")
        grades_data = fetch_json(page, INFINITE_CAMPUS_GRADES_API)

        # Step 6: Fetch assignments JSON
        print("  Fetching assignments from IC API...")
        assignments_data = fetch_json(page, INFINITE_CAMPUS_ASSIGN_API)

        browser.close()
        return grades_data, assignments_data

def parse_grades(data, target_term):
    """Extract courses and grades for the target term."""
    courses = []
    if not data:
        return courses

    def find_terms(obj):
        if isinstance(obj, list):
            for item in obj:
                yield from find_terms(item)
        elif isinstance(obj, dict):
            if "termName" in obj and "courses" in obj:
                yield obj
            else:
                for v in obj.values():
                    yield from find_terms(v)

    seen = set()
    for term in find_terms(data):
        if term.get("termName", "").strip() != target_term:
            continue
        for course in term.get("courses", []):
            name = course.get("courseName", "Unknown")
            teacher = course.get("teacherDisplay", "")
            grade, percent = "N/A", ""
            for task in course.get("gradingTasks", []):
                if task.get("taskName") == "Trimester Grade":
                    score = task.get("progressScore") or task.get("score", "")
                    pct = task.get("progressPercent") or task.get("percent")
                    if score:
                        grade = score
                    if pct is not None:
                        percent = f"{pct:.2f}%"
                    break
            key = f"{name}-{teacher}"
            if key not in seen:
                seen.add(key)
                courses.append({"name": name, "teacher": teacher, "grade": grade, "percent": percent})

    courses.sort(key=lambda c: c["name"])
    return courses


# ── ASSIGNMENTS ───────────────────────────────────────────

def parse_assignments(data, days_back):
    """Extract assignments from the last N days, grouped by date."""
    grouped = {}
    if not data:
        return grouped

    cutoff = datetime.now() - timedelta(days=days_back)

    for item in (data if isinstance(data, list) else []):
        name = item.get("assignmentName", "Unknown")
        course = item.get("courseName", "")
        due_raw = item.get("dueDate") or ""
        score = item.get("score")
        total = item.get("totalPoints")
        missing = item.get("missing") or False
        comments = item.get("comments") or ""

        # Parse ISO date like "2026-03-05T04:59:00.000Z"
        due_dt = None
        try:
            due_dt = datetime.strptime(due_raw[:19], "%Y-%m-%dT%H:%M:%S") - timedelta(hours=5)
        except Exception:
            continue

        if due_dt < cutoff:
            continue

        date_str = due_dt.strftime("%A %m/%d/%Y")
        if date_str not in grouped:
            grouped[date_str] = []

        score_str = ""
        if score is not None and total is not None:
            score_str = f"{score}/{total}"
        elif score is not None:
            score_str = str(score)

        grouped[date_str].append({
            "name": name,
            "course": course,
            "score": score_str,
            "missing": missing,
            "comments": comments,
        })

    # Sort dates chronologically
    sorted_grouped = dict(sorted(
        grouped.items(),
        key=lambda x: datetime.strptime(x[0], "%A %m/%d/%Y")
    ))
    return sorted_grouped
# ── PRINT ─────────────────────────────────────────────────

def print_summary(courses, assignments, target_term, days_back):
    print("\n" + "=" * 60)
    print(f"INFINITE CAMPUS SUMMARY — {target_term}")
    print(f"Generated: {datetime.now().strftime('%A, %B %d %Y at %I:%M %p')}")
    print("=" * 60)

    print(f"\n── GRADES ({target_term}) ──────────────────────────────────")
    if not courses:
        print("  No grades found.")
    else:
        for c in courses:
            grade_str = c["grade"] + (f"  ({c['percent']})" if c["percent"] else "")
            print(f"  {c['name']:<35} {grade_str}")
            print(f"    Teacher: {c['teacher']}")

    print(f"\n── ASSIGNMENTS (last {days_back} days) ──────────────────────")
    if not assignments:
        print("  No assignments found.")
    else:
        for date, items in assignments.items():
            print(f"\n  {date}")
            for a in items:
                missing_tag = "  [MISSING]" if a["missing"] else ""
                score_tag = f"  {a['score']}" if a["score"] else ""
                print(f"    • {a['name']}")
                print(f"      {a['course']}{score_tag}{missing_tag}")
                if a["comments"]:
                    print(f"      Note: {a['comments']}")

    print("\n" + "=" * 60)


# ── EMAIL ─────────────────────────────────────────────────

def build_email_body(courses, assignments, target_term, days_back):
    lines = []
    lines.append(f"INFINITE CAMPUS SUMMARY — {target_term}")
    lines.append(f"Generated: {datetime.now().strftime('%A, %B %d %Y at %I:%M %p')}")
    lines.append("=" * 50)

    lines.append(f"\nGRADES ({target_term})")
    lines.append("-" * 30)
    if not courses:
        lines.append("No grades found.")
    else:
        for c in courses:
            grade_str = c["grade"] + (f"  ({c['percent']})" if c["percent"] else "")
            lines.append(f"{c['name']}")
            lines.append(f"  Teacher: {c['teacher']}")
            lines.append(f"  Grade:   {grade_str}")

    lines.append(f"\nASSIGNMENTS (last {days_back} days)")
    lines.append("-" * 30)
    if not assignments:
        lines.append("No assignments found.")
    else:
        for date, items in assignments.items():
            lines.append(f"\n{date}")
            for a in items:
                missing_tag = "  [MISSING]" if a["missing"] else ""
                score_tag = f"  {a['score']}" if a["score"] else ""
                lines.append(f"  - {a['name']}")
                lines.append(f"    {a['course']}{score_tag}{missing_tag}")
            if a["comments"]:
                lines.append(f"    Note: {a['comments']}")

    lines.append("\n" + "=" * 50)
    lines.append("Sent by your Infinite Campus Script")
    return "\n".join(lines)


def send_email(body, target_term):
    print("\nSending email...")
    msg = MIMEMultipart()
    msg["From"] = INFINITE_CAMPUS_FROM_EMAIL
    msg["To"] = INFINITE_CAMPUS_EMAIL_RECIPIENTS
    msg["Subject"] = f"Jackson - Grades and Assignments ({target_term}) — {datetime.now().strftime('%b %d, %Y')}"
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(INFINITE_CAMPUS_FROM_EMAIL, INFINITE_CAMPUS_FROM_PASSWORD)
            server.sendmail(INFINITE_CAMPUS_FROM_EMAIL, [e.strip() for e in INFINITE_CAMPUS_EMAIL_RECIPIENTS.split(",")], msg.as_string())
        print(f"Email sent to {INFINITE_CAMPUS_EMAIL_RECIPIENTS}!")
    except smtplib.SMTPAuthenticationError:
        print("Gmail login failed. Make sure you are using an App Password, not your regular password.")
    except Exception as e:
        print(f"Failed to send email: {e}")


# ── MAIN ──────────────────────────────────────────────────


def build_json_output(courses, assignments, target_term):
    """Build a clean JSON structure suitable for feeding to an LLM."""
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "term": target_term,
        "grades": [
            {
                "course": c["name"],
                "teacher": c["teacher"],
                "grade": c["grade"],
                "percent": c["percent"],
            }
            for c in courses
        ],
        "assignments": [
            {
                "date": date,
                "items": [
                    {
                        "name": a["name"],
                        "course": a["course"],
                        "score": a["score"],
                        "missing": a["missing"],
                        "comments": a.get("comments", ""),
                    }
                    for a in items
                ],
            }
            for date, items in assignments.items()
        ],
    }


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch your Infinite Campus grades and assignments."
    )

    # Output mode — must pick exactly one
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--print",
        action="store_true",
        dest="do_print",
        help="Print a human-readable summary to the terminal.",
    )
    group.add_argument(
        "--email",
        action="store_true",
        help="Send the summary by email (no terminal output).",
    )
    group.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout and save ic_output.json (for LLM use).",
    )

    # Optional filters with defaults from config
    parser.add_argument(
        "--term",
        default=TARGET_TERM,
        help=f"Which term to show grades for (default: {TARGET_TERM}). E.g. T1, T2, T3.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=ASSIGNMENT_DAYS,
        help=f"How many days back to show assignments (default: {ASSIGNMENT_DAYS}).",
    )

    args = parser.parse_args()

    grades_data, assignments_data = login_and_fetch()
    courses = parse_grades(grades_data, args.term)
    assignments = parse_assignments(assignments_data, args.days)

    if args.json:
        output = build_json_output(courses, assignments, args.term)
        print(json.dumps(output, indent=2))

        # For now, printing to the file is disabled
        #json_path = "ic_output.json"
        #with open(json_path, "w") as f:
        #    json.dump(output, f, indent=2)
        #print(f"\n(JSON also saved to {json_path})", file=sys.stderr)

    elif args.do_print:
        print_summary(courses, assignments, args.term, args.days)

    elif args.email:
        email_body = build_email_body(courses, assignments, args.term, args.days)
        send_email(email_body, args.term)


if __name__ == "__main__":
    main()