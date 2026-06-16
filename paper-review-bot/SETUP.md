# Setup — Daily BraTS-MET Paper Review (runs on GitHub Actions, no Mac needed)

This folder contains everything needed to run the daily paper review fully in
the cloud. Once set up, it runs every day at 6:00 AM IST on GitHub's servers —
your Mac and the Claude app don't need to be open at all.

## What it does each day
1. Calls the Anthropic API (Claude) with web search enabled to find up to 10 new
   papers on brain metastases segmentation / BraTS-MET / relevant architectures.
2. Reviews each paper's core technique and gives an implementation verdict.
3. Appends the result to `PAPER_REVIEWS.md` in this repo (committed automatically).
4. Emails you the full review via Gmail.

## One-time setup (about 10 minutes)

### 1. Create a GitHub repo
Create a new **private** repository (e.g. `brats-met-paper-bot`) at github.com/new.

### 2. Add these three files to the repo
Copy them exactly as they are in this folder, preserving the path:
- `daily_paper_review.py` (repo root)
- `requirements.txt` (repo root)
- `.github/workflows/daily-paper-review.yml` (must be in this exact folder path)

You can do this by dragging the files into the GitHub web UI ("Add file" →
"Upload files"), or via git from your terminal:
```bash
cd brats-met-paper-bot
git init
git add .
git commit -m "Initial setup"
git branch -M main
git remote add origin https://github.com/<your-username>/brats-met-paper-bot.git
git push -u origin main
```

### 3. Get an Anthropic API key
Go to https://console.anthropic.com → Settings → API Keys → Create Key.
Note: this is billed separately from any Claude subscription (pay-as-you-go,
you'll need to add billing/credits to the API console). Cost for this workload
is small — roughly a few cents per day (web search is $10 per 1,000 searches,
plus token costs for ~10 paper summaries).

### 4. Generate a Gmail App Password
- Make sure 2-Step Verification is ON for your Google account (Google Account →
  Security → 2-Step Verification).
- Go to https://myaccount.google.com/apppasswords
- Create an app password for "Mail" — copy the 16-character code shown.
  (This is NOT your normal Gmail password — don't use that.)

### 5. Add repo secrets
In your GitHub repo: Settings → Secrets and variables → Actions → "New repository secret".
Add all four:
| Secret name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | the key from step 3 |
| `GMAIL_ADDRESS` | the Gmail address you generated the app password for |
| `GMAIL_APP_PASSWORD` | the 16-character app password from step 4 |
| `RECIPIENT_EMAIL` | `jangirvineet2@gmail.com` (can be the same as GMAIL_ADDRESS) |

### 6. Test it
Go to the repo's "Actions" tab → "Daily BraTS-MET Paper Review" workflow →
"Run workflow" button → run it manually once. Check:
- The Action run succeeds (green check).
- `PAPER_REVIEWS.md` in the repo gets a new dated section.
- An email arrives in your inbox.

That's it — after this it runs automatically every day at 6:00 AM IST
(00:30 UTC), entirely on GitHub's infrastructure.

## Notes
- GitHub's scheduled workflows can run a few minutes late during high load —
  this is normal and not something to debug.
- If you ever want to change the time, edit the `cron` line in
  `.github/workflows/daily-paper-review.yml` (cron is in UTC, IST = UTC+5:30).
- If you'd rather not pay for Anthropic API usage, the existing Cowork
  scheduled task (the one tied to your Mac/Claude app) still works as a free
  fallback — the two can coexist.
