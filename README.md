# The Daily Telegram

A local-only Telegram channel intelligence briefing system. Monitors Telegram channels, analyses posts with a locally-running Vision-Language Model via LM Studio, and generates daily ranked PDF briefings with an AI-synthesised intelligence front page — no cloud APIs, no data leaves your machine.

## How it works

```
Telegram channels
       ↓  (Telethon)
   scraper.py  →  SQLite (posts)
                      ↓
   analyzer.py  ←  LM Studio VLM
   (title, scores, category, threat_level, key entities)
                      ↓
     triage.py  (composite score × channel priority × credibility, keyword boost, rumour
                 penalty, recency decay, story clustering with corroboration boost,
                 main/appendix split)
                      ↓
   generator.py  →  briefings/YYYY-MM-DD/TheDailyTelegram_YYYY-MM-DD_HHMMSS.pdf
                      ↓
  synthesiser.py  ←  LM Studio (intelligence synthesis)
   (triaged main items + 7-day mention trends + yesterday's themes →
    situation summary, key themes with citations & continuity, signals & warnings,
    named actors, emerging actors)
                      ↓
   prepends intelligence front page to briefing PDF
```

Two operating modes:
- **`--batch`**: one-shot run — scrape all channels, analyse everything, generate today's PDF with intelligence front page prepended automatically
- **`--daemon`**: long-running — listen for live messages, analyse as they arrive, generate PDF daily at a configured time

---

## Prerequisites

### 1. Python 3.11+

```bash
python --version   # must be 3.11 or newer
```

### 2. Telegram API credentials

1. Go to [https://my.telegram.org/apps](https://my.telegram.org/apps) and log in
2. Create a new application (any name)
3. Note your **API ID** (integer) and **API Hash** (hex string)

### 3. LM Studio

Download from [https://lmstudio.ai](https://lmstudio.ai) and install it. You need a Vision-Language Model (VLM) loaded — one that can analyse images alongside text.

Recommended model (small and capable):
- `google/gemma-4-12b` — default model, excellent intelligence and vision capabilities
- Alternatively you can try a larger GGUF model with vision support

To start the server inside LM Studio: **Local Server → Start Server** (default port 1234).

---

## Installation

```bash
# Clone the repository, then:
cd TheDailyTelegram

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate          # Linux / macOS
# or: .venv\Scripts\activate       # Windows

# Install the package and all dependencies
pip install -e ".[dev]"
```

Verify the install:
```bash
python -m tg_compiler.main --help
```

Expected output:
```
usage: tg_compiler [-h] [--config CONFIG] [--batch] [--daemon] [--generate] [--analyse] [--since TIME] [--layout {desktop,mobile}]
```

---

## Configuration

### Step 1 — Copy the example config

```bash
cp config.yaml.example config.yaml
```

### Step 2 — Fill in your credentials

Open `config.yaml` and edit:

```yaml
telegram:
  api_id: 123456          # your integer API ID from my.telegram.org — or use TG_API_ID env var
  api_hash: "abc123..."   # your API hash from my.telegram.org — or use TG_API_HASH env var
  session_name: "briefing_session"   # filename for the Telegram session (no extension needed)
  channels:
    - username: "@channelname"   # public @username, or use id: 123456789 for private channels
      slug: "news"               # short label used in file paths and briefing headings
      priority: 1.0              # composite score multiplier (0.1–2.0); higher = ranked more prominently
      credibility: 1.0           # channel credibility prior (0.1–2.0), also multiplied into the score
      # custom_prompt: |         # optional: override the LLM system prompt for this channel only
      #   You are an analyst specialised in...
    - username: "@another_channel"
      slug: "intel"
      priority: 0.8
  rate_limit_delay_ms: 500      # pause between channel scrapes in ms (be conservative with Telegram)
  lookback_seconds: 604800      # how far back to fetch on first run (default: 1 week)
                                # use --batch --since HH:MM for a one-off lookback instead

lmstudio:
  model: "google/gemma-4-12b-qat"  # must match the model name shown in LM Studio
  server_host: "localhost"          # IP or hostname if LM Studio is on another machine
  server_port: 1234
  # api_token: "lms-..."           # optional; overridden by LM_API_TOKEN env var
  temperature: 0.3
  max_tokens: 800
  max_concurrent_analyses: 1       # parallel LLM calls; increase if your GPU can handle it

triage:
  keywords: ["urgent", "breaking", "launch"]  # words that add keyword_boost to a post's score
  keyword_boost: 0.5        # score added when a keyword matches (total capped at 5.0)
  min_composite_score: 3.5  # posts below this go to the Appendix section
  min_main_items: 15        # if fewer posts clear the threshold, top appendix items are promoted to fill
  max_main_items: 50        # hard cap on main briefing length (overflow → appendix)
  dedup_window_secs: 7200   # time window for entity-overlap deduplication (default: 2h)
  dedup_summary_window_secs: 21600   # window for summary/title word-overlap dedup (default: 6h)
  entity_cluster_window_secs: 86400  # wider dedup window for entity-cluster matching (default: 24h)
  dedup_jaccard_threshold: 0.28      # min word-overlap ratio for summary/title dedup
  dedup_entity_overlap_count: 3      # shared entities required within dedup_window_secs
  dedup_entity_cluster_overlap_count: 4  # shared entities required within entity_cluster_window_secs
  recency_half_life_hours: 24.0      # composite score halves every this many hours of post age
  recency_floor: 0.7                 # minimum recency multiplier, however old the post
  threat_multipliers:                # ranking multiplier per threat badge
    CRITICAL: 1.15
    HIGH: 1.05
    MODERATE: 1.0
    LOW: 0.85
  corroboration_weight: 0.15         # score multiplier added per corroborating channel
  corroboration_cap: 1.5             # max total multiplier from corroboration boost
  rumor_penalty: 0.7                 # score multiplier applied to posts categorised "Rumor"

generation:
  output_dir: "./briefings"   # where PDFs and markdown are saved
  generate_at: "23:59"        # daily auto-generation time in daemon mode (HH:MM, in timezone below)
  timezone: "UTC"             # IANA timezone for generate_at (e.g. "Europe/London")
  pdf_layout: "desktop"       # PDF CSS layout: "desktop" or "mobile" (override with --layout)
  # share_to_directory: "/path/to/shared/folder"  # if set, copy the final PDF here after generation
  # Intelligence Assessment coverage is governed by triage.max_main_items

storage:
  db_path: "./data/briefing.db"
  media_dir: "./data/media"
  retention_days: 30          # delete media older than this many days
```

### Step 3 — Set up your `.env` file

The app loads `.env` automatically at startup. Copy the example and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
TG_API_ID=123456              # overrides telegram.api_id in config.yaml
TG_API_HASH=your_api_hash     # overrides telegram.api_hash in config.yaml
LM_API_TOKEN=your_api_token_here # required if LM Studio has authentication enabled
```

`TG_API_ID` and `TG_API_HASH` override the corresponding YAML fields. `LM_API_TOKEN` is only needed if you have enabled API token authentication in LM Studio's settings.

If LM Studio is on a different machine, set `server_host` in `config.yaml` to its IP address (e.g. `192.168.1.96`).

---

## First run — Telegram authentication

The first time you run any command that connects to Telegram, Telethon will prompt you to log in interactively. This only happens once; the session is saved to `<session_name>.session`.

```bash
source .venv/bin/activate
python -m tg_compiler.main --batch
```

You will see:
```
Please enter your phone number (or bot token):
```

1. Enter your phone number in international format (e.g. `+447700900000`)
2. Telegram will send a login code to your Telegram app
3. Enter the code when prompted
4. If you have 2FA enabled, enter your password

After successful login, `briefing_session.session` is created and subsequent runs connect silently.

---

## Batch mode — one-shot scrape and report

Batch mode scrapes all configured channels, analyses every unseen post, generates today's PDF, and automatically prepends an AI-synthesised intelligence front page.

```bash
source .venv/bin/activate
python -m tg_compiler.main --batch
```

What happens:
1. Connects to Telegram
2. For each channel: fetches every message since the last run (no cap), downloads attached photos — a failure on one channel is logged and the rest continue
3. Sends each new post to LM Studio for analysis — headline title, importance, urgency, credibility, relevance, category, threat level, and key entities. Posts with fewer than 30 characters of text and no media are skipped without an LLM call
4. Runs triage: scores posts (channel priority × credibility, keyword boost, rumour penalty, recency decay), clusters cross-channel reports of the same story — the best report is kept and the rest become "corroborated by" references that boost its score — then splits into main/appendix
5. Generates `briefings/YYYY-MM-DD/TheDailyTelegram_YYYY-MM-DD_HHMMSS.pdf`
6. Sends the triaged main items, a 7-day entity/category mention-trend table, and yesterday's assessment themes to LM Studio for intelligence synthesis
7. Prepends a structured intelligence front page (situation summary, key themes with source citations and continuity tags, signals & warnings, named actors, emerging actors) to the briefing PDF, and persists the assessment for next-day continuity
8. Disconnects

Subsequent `--batch` runs on the same day are safe — cursor tracking ensures no post is fetched twice, and UNIQUE constraints prevent duplicate DB entries. If LM Studio is unreachable during the front page step, a warning is logged and the briefing PDF is kept as-is.

Typical log output:
```
2026-06-08 09:00:01 INFO Scraped 14 new posts from news
2026-06-08 09:00:02 INFO Scraped 3 new posts from intel
2026-06-08 09:00:45 INFO Analysed 15 posts (skipped 2)
2026-06-08 09:00:46 INFO Briefing generated: briefings/2026-06-08/TheDailyTelegram_2026-06-08_090046.pdf
2026-06-08 09:00:47 INFO Synthesising intelligence assessment from 17 posts…
2026-06-08 09:01:30 INFO Intelligence front page prepended → briefings/2026-06-08/TheDailyTelegram_2026-06-08_090046.pdf
```

### Re-scraping from a specific time — `--since`

To fetch posts from a point further back than the last run, use `--since`. This automatically resets channel cursors and sets the lookback window — no manual config edits needed.

```bash
# Re-scrape from midnight UTC today
python -m tg_compiler.main --batch --since 00:00

# Re-scrape from the start of a specific date
python -m tg_compiler.main --batch --since 2026-06-01

# Re-scrape from a specific date and time
python -m tg_compiler.main --batch --since 2026-06-07T06:00
```

Accepted formats: `HH:MM` (today at that UTC time), `YYYY-MM-DD` (midnight on that date), `YYYY-MM-DDTHH:MM` (exact UTC datetime).

**`--since` resets channel cursors** so the scraper re-fetches from Telegram. Already-seen posts hit the `UNIQUE(channel_id, message_id)` constraint and are silently discarded — no duplicate DB entries. Already-analysed posts are skipped by `get_unanalysed_posts()` — no LLM calls are wasted. The downside is Telegram still has to serve those message pages, which wastes API quota.

> **Use `--since` only when you intentionally need a historical lookback.** For routine same-day re-runs, use plain `--batch` — it uses the cursor and fetches only messages that arrived since the last run.

---

## Daemon mode — live monitoring

Daemon mode runs indefinitely, listening for new messages in real time and generating a PDF automatically at the configured `generate_at` time each day.

> **Important:** The daemon is a live listener only. It processes messages that arrive while it is running — it does **not** backfill historical posts. Always run `--batch` first to catch up on any posts you want in the briefing, then start the daemon.

```bash
# Recommended startup sequence:
python -m tg_compiler.main --batch   # catch up on history first
python -m tg_compiler.main --daemon  # then switch to live monitoring
```

### Start LM Studio first

Ensure LM Studio is running with a model loaded before starting the daemon:
- Open LM Studio → Local Server tab → click **Start Server**
- Confirm the server shows "Running on port 1234"

### Start the daemon

```bash
source .venv/bin/activate
python -m tg_compiler.main --daemon
```

What happens at startup:
1. Opens a Telegram client session
2. Resolves all configured channels and registers a live message listener
3. Spawns a background scheduler for daily briefing generation
4. Logs: `Daemon running on N channels`

What happens when a new message arrives:
1. Downloads attached media (if any)
2. Inserts a `PostRecord` into SQLite
3. Sends the post to LM Studio for analysis (including threat level) and saves the result
4. Duplicate posts (by channel_id + message_id) are silently skipped

What happens at `generate_at` time each day:
1. Runs triage on all posts analysed that day
2. Generates `briefings/YYYY-MM-DD/TheDailyTelegram_YYYY-MM-DD_HHMMSS.pdf`
3. Synthesises and prepends the intelligence front page
4. Purges media directories older than `retention_days`

### Stopping the daemon

Press `Ctrl+C`. The Telegram client disconnects cleanly.

### Running as a background service (optional)

```bash
# Using nohup
nohup python -m tg_compiler.main --daemon > logs/daemon.log 2>&1 &
echo $! > daemon.pid

# Stop it later
kill $(cat daemon.pid)
```

Or use a systemd service unit if running on Linux.

---

## Generating a report manually

If you have posts in the database but want to regenerate today's report without re-scraping:

```bash
source .venv/bin/activate
python -m tg_compiler.main --generate
```

To prepend the intelligence front page to an existing briefing PDF (e.g. after a `--generate`):

```bash
python -m tg_compiler.main --analyse
# or for a specific date:
python -m tg_compiler.main --analyse --since 2026-06-07
```

`--analyse` finds the most recent `TheDailyTelegram_*.pdf` in the date subdirectory, re-runs triage to reconstruct the same main-item set as the briefing (recency decay is anchored to that day, so past dates rank identically), synthesises via LM Studio, and prepends the front page. Re-running it replaces the existing front page rather than stacking a second one. Under `--batch` this runs automatically, so `--analyse` is mainly useful after a standalone `--generate`.

### Mobile layout — `--layout`

By default the PDF uses the desktop CSS layout. Pass `--layout mobile` (with `--batch`, `--generate`, or `--daemon`) to use a layout with larger text, tighter margins, and a single-column appendix, optimised for reading on a phone:

```bash
python -m tg_compiler.main --generate --layout mobile
```

The default can also be set permanently via `generation.pdf_layout` in `config.yaml` (`"desktop"` or `"mobile"`); `--layout` overrides it for a single run.

### Sharing the PDF — `share_to_directory`

Set `generation.share_to_directory` in `config.yaml` to a directory path, and the final generated PDF (after the intelligence front page has been prepended) is copied there too — e.g. a Syncthing/Dropbox/Nextcloud folder for reading on another device. Leave it unset (the default) to disable this.

---

## Reading the reports

Reports are written to date-named subdirectories under `./briefings/` (configurable via `generation.output_dir`).

```
briefings/
└── 2026-06-08/
    ├── TheDailyTelegram_2026-06-08_090046.pdf   ← primary report (with front page)
    └── briefing_2026-06-08.md                      ← Markdown source
```

Each `--batch` or `--generate` run writes a new uniquely timestamped PDF. The `.md` file is the source of truth and is overwritten on each run.

### Report structure

**Intelligence Front Page** — prepended automatically. Contains:
- *Situation Summary* — 3-5 sentence analyst overview of the day's geopolitical picture, informed by 7-day mention trends
- *Key Themes* — 3-5 cross-cutting patterns across today's reports, each with source citations (channel + time + link) and a continuity tag (*confirmed* / *escalating* / *retired*) relative to yesterday's assessment
- *Signals & Warnings* — 3-5 developments to watch with observable indicators, each with source citations
- *Named Actors* — 4-6 most significant actors and their activity today
- *Emerging Actors / Topics* — entities mentioned today but absent from the prior 7 days (shown once a baseline exists)

**Lead Reports** — the day's most important stories in full detail. Every CRITICAL-rated item is guaranteed a slot regardless of its score (even one that scored into In Brief), topped up with the highest-scoring remaining reports to at least 10 — never truncating criticals, so the section grows when criticals alone exceed 10. Cross-channel reports of the same story (detected by word overlap, or shared named entities with alias normalisation so "U.S."/"US"/"United States" match) are clustered: the highest-scoring report appears, with a **"Corroborated by N other channels"** line linking to the duplicates (N counts distinct other channels; repeat posts from the story's own channel are listed separately as "Related posts from this channel" and don't inflate the count), and cross-channel corroboration boosts the story's score. Each entry shows:
- **Threat level badge**: ■ CRITICAL (red) · ■ HIGH (orange) · ■ MODERATE (amber) · ■ LOW (green)
- **Category** in backtick style: `` `Breaking News` `` / `` `Analysis` `` / `` `Official Statement` `` / `` `Rumor` `` / `` `Media` `` / `` `Other` ``
- LLM-generated headline title (5-10 words)
- Channel, post timestamp, and direct link to the original Telegram post (↗ t.me)
- Full summary from the VLM
- Composite score out of 5
- Key named entities
- Image analysis excerpt (if the post had a substantive image)
- Attached images (up to 3, embedded in PDF)

**Other Developments** — the remaining reports that cleared `min_composite_score`, sorted by composite score descending. Each entry is a compact block: badge, category, bold headline and channel/time/score/link meta, then the full summary and (when present) a "Corroborated by N other channels" line with links. If fewer than `min_main_items` clear the threshold, the highest-scoring remainder are promoted, and the total is capped at `max_main_items` (excess goes to In Brief).

**In Brief** — lower-priority posts, listed compactly with direct Telegram links; cross-channel corroboration is abbreviated to a `+N corrob` marker.

Each story appears in exactly one section — lead stories are not repeated below.

**Statistics** — a compact block with the published item count, lead/other/in-brief split, channels covered, a per-category breakdown, and (after a `--batch` run) the pipeline funnel: scraped → analysed → skipped (low-content) → duplicates merged.

**Reader's Key** — static smallprint at the end of every edition explaining how the document is produced, its section order, the scoring formula (including the configured score threshold), de-duplication, and threat levels. It is template boilerplate, never written or altered by the LLM.

### Threat level scale

| Badge | Level | Meaning |
|---|---|---|
| ■ red | CRITICAL | Imminent risk of mass casualties, confirmed state-level military action underway, nuclear/chemical/biological threat, or active attack on critical infrastructure |
| ■ orange | HIGH | Confirmed armed conflict development, significant political crisis, major terror attack, or credible escalation warning from a named senior state official |
| ■ amber | MODERATE | Ongoing conflict updates, diplomatic developments, significant arrests or detentions, or unverified but plausible escalation claims |
| ■ green | LOW | Background context, routine troop movement reports, unverified rumours, social media content, statistical or historical reports |

### Composite scoring formula

```
base  = 0.4 × importance + 0.3 × urgency + 0.2 × credibility + 0.1 × relevance
score = base × channel_priority × channel_credibility        (capped at 5.0 after keyword boost)
      × rumor_penalty (if category is "Rumor")
      × threat_multiplier (CRITICAL 1.15 / HIGH 1.05 / MODERATE 1.0 / LOW 0.85)
      × recency multiplier (halves every recency_half_life_hours, floored at recency_floor)
      × corroboration boost (1 + 0.15 per corroborating channel, capped at 1.5×)
```

Each dimension is rated 1–5 by the VLM. Keyword matches add `keyword_boost` (default 0.5) before the cap. Badge guard: a CRITICAL threat level on a Rumor-category post or one with credibility ≤ 2 is demoted to HIGH at triage time — CRITICAL implies confirmation, and unsubstantiated posts must not claim guaranteed Lead Reports slots. The recency decay is anchored to the briefing day, so regenerating a past date reproduces that day's ranking. Displayed scores are clamped to 5.0.

---

## Troubleshooting

**"No module named tg_compiler"**  
The virtual environment is not active. Run `source .venv/bin/activate` first.

**"LM Studio is not reachable" / connection refused on port 1234**  
LM Studio server is not running, or `server_host`/`server_port` in `config.yaml` don't match. Start it via LM Studio → Local Server → Start Server. If LM Studio runs on another machine, set `lmstudio.server_host` to its IP address. The app uses LM Studio's OpenAI-compatible HTTP endpoint (`/v1/chat/completions`) — ensure "Enable API server" is on.

**"ChannelPrivateError"**  
Your Telegram account is not a member of that channel. Join it in the Telegram app and retry.

**"FloodWaitError: X seconds"**  
Telegram rate-limited the request. The scraper will pause automatically and resume. If it happens often, increase `rate_limit_delay_ms`.

**"ValidationError: extra inputs are not permitted"**  
A field in `config.yaml` is misspelled or unknown. Check the field name against `config.yaml.example`.

**"TG_API_ID env var must be an integer"**  
The `TG_API_ID` environment variable is set but contains a non-integer value. Either unset it or fix the value.

**PDF is empty or has no posts**  
Either no posts were scraped today, or LM Studio analysis has not run yet. Run `--batch` to trigger a full scrape+analyse cycle, then check `--generate`.

**Intelligence front page not prepended**  
If LM Studio was unreachable during the synthesis step, a warning is logged and the briefing is kept as-is. Check LM Studio is running and retry with `--analyse`.

**Session file issues after moving the project**  
Delete `<session_name>.session` and re-authenticate by running `--batch` again.

---

## Running tests

```bash
source .venv/bin/activate
pytest                          # all tests
pytest tests/test_db.py -v      # single file
pytest tests/test_triage.py::test_composite_score_formula -v   # single test
```

Tests use in-memory SQLite and do not require Telegram credentials or a running LM Studio server.
