# Demo video script (≈ 8 minutes)

Record the screen (OBS Studio or Windows Game Bar `Win+Alt+R`, both free). Use the live site
**https://signal-stack-kohl.vercel.app** for the walkthrough of the three finished companies, and a local run
(`uvicorn signalstack.web:app --port 8000` with `GROQ_API_KEY` in `.env`) for the live "add a company" part, because a full
discovery + crawl takes longer than Vercel's free 300 s request limit.

| Time | Show | Say |
|---|---|---|
| 0:00 | Home page | "Signal Stack turns a company into a continuously monitored business: research → interview → Signal Stack → monitoring → Monday brief. Everything is free/open-source; the AI runs on open-weights models on Groq's free tier." |
| 0:30 | Add **MeetUniversity** + `https://www.meetuniversity.com`; Overview shows research running | "Research reads the site (robots.txt respected), sitemap, Wikipedia, news and App Store presence." |
| 1:15 | **Intelligence Profile** → facts vs assumptions | "Facts are only facts if they match text on a fetched page — each has its source link. Inferences are assumptions until the founder confirms." |
| 2:00 | **AI Interview** → answer 2 questions live, show a follow-up appear, skip one | "Questions come from this company's research. It asks one at a time, remembers answers, asks follow-ups, adapts the next question, and I can skip confidential ones and edit later." |
| 3:00 | **Signal Stack** → four columns + competitor table | "Competitors are discovered and verified by fetching their homepages — nothing is hard-coded. Every source has a reason written from probe evidence: specificity, recency, credibility." |
| 3:45 | **Source Manager** → filter blocked → Reddit row; approve a discovered source; add a source; upload the synthetic CRM CSV | "Access-restricted sources like Reddit are identified but never scraped. Internal data: CSV, Excel, PDF, text." |
| 4:30 | **Intelligence Feed** → filter competitors / threat / primary source; open one item | "Evidence on the left, interpretation on the right. Reliability, evidence strength and relevance are separate scores; the model's confidence is not evidence." |
| 5:30 | **Monday Brief** → scroll the 10 sections, click a signal ID into the appendix, change an action status, export email HTML | "Every claim cites signals; empty categories say so. Actions have owners and are tracked." |
| 6:45 | **Settings** + **Runs & Failures** + **Source-Health Audit** | "Weekly brief and monthly audit run on their own schedules; failures and missed runs are visible." |
| 7:15 | Switch to **Zerodha** and **Licious** → different interview questions, competitors and briefs; Licious **context-change revision** | "Changing the company changes everything. After a context change the stack is reassessed and the system explains what changed." |
| 8:00 | Close | "Repo, reports, docs and cost statement are linked in the README." |
