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
- **`--daemon`**: long-running — listen for live messages, store them and analyse in a sweep every 10 minutes, generate PDF daily at a configured time

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

The pipeline can use **one model for everything, or a different model per stage**
(see `analysis_model` / `synthesis_model` below). The two stages want different
things:

- **Analysis** scores every scraped post — a thousand or more a day. It wants a
  small, fast, *non-reasoning* VLM. A reasoning model spends most of its output
  budget deliberating (measured: 3,515 reasoning tokens per post against ~250
  tokens of actual JSON), which makes a day's analysis take many hours.
- **Synthesis** writes the intelligence front page once a day from the already
  triaged posts. Reasoning genuinely helps here and costs one call.

Recommended pairing:
- Analysis: `mistralai/ministral-3-3b` (3B + 0.4B vision encoder, ~1-3 s/post)
- Synthesis: any larger reasoning model you like

A single capable VLM such as `google/gemma-4-12b-qat` also works for both — just
leave `analysis_model` and `synthesis_model` unset.

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
usage: tg_compiler [-h] [--config CONFIG] [--batch] [--daemon] [--generate]
                   [--analyse] [--since TIME] [--analysis-profile NAME]
                   [--layout {desktop,mobile}]
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
                                # capped at storage.analysis_lookback_days; --since overrides
                                # use --batch --since HH:MM for a one-off lookback instead

lmstudio:
  model: "google/gemma-4-12b-qat"  # must match the model name shown in LM Studio
  server_host: "localhost"          # IP or hostname if LM Studio is on another machine
  server_port: 1234
  # api_token: "lms-..."           # optional; overridden by LM_API_TOKEN env var
  temperature: 0.3
  # Per-post analysis token budget (replaces the old fixed max_tokens):
  #   budget = min(analysis_base_tokens + text_chars * analysis_tokens_per_char
  #                + images * analysis_tokens_per_image, analysis_max_tokens)
  # Short posts stay cheap; long/image-heavy posts get headroom. The base is generous
  # because reasoning models spend ~800+ tokens deliberating before emitting JSON, and
  # LM Studio's OpenAI-compatible API has no separate reasoning-token control.
  analysis_base_tokens: 1500       # flat floor: reasoning deliberation + structured JSON output
  analysis_tokens_per_char: 0.3    # extra budget per character of prompt text (truncated at 3000 chars)
  analysis_tokens_per_image: 250   # extra budget per attached image (max 3 per post)
  analysis_max_tokens: 4000        # hard ceiling on any single analysis call
  synthesis_max_tokens: 24000      # token budget for the intel front-page synthesis (keep generous: reasoning models deliberate before emitting JSON)
  max_concurrent_analyses: 4       # parallel LLM calls; see "Throughput" below

  # --- Per-stage models (optional; both fall back to `model`) ---
  analysis_model: "mistralai/ministral-3-3b"   # fast non-reasoning VLM for scoring every post
  synthesis_model: "prism-ml/bonsai-27b"       # reasoning model for the daily front page
  manage_models: true              # load/unload each stage's model via the LM Studio SDK
  unload_others: true              # free VRAM before loading the next stage's model
  model_ttl_seconds: 3600          # LM Studio unloads an idle model after this
  # model_context_length: 32768    # optional load-time context override

  # --- Batched analysis (optional; 1 = one post per call, the default) ---
  batch_size: 1                    # text-only posts per LLM call
  batch_size_with_images: 1        # media-bearing posts per LLM call
  # Only worth raising for a *reasoning* analysis model — see "Throughput" below.

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
  # WARNING: the briefing date is always the UTC calendar date, regardless of
  # this timezone. Choose a generate_at whose UTC equivalent is late in the day
  # being reported — a local time that converts to just after UTC midnight
  # compiles the *new* (near-empty) UTC day instead of the day you meant.
  timezone: "UTC"             # IANA timezone for generate_at (e.g. "Europe/London")
  pdf_layout: "desktop"       # PDF CSS layout: "desktop" or "mobile" (override with --layout)
  # share_to_directory: "/path/to/shared/folder"  # if set, copy the final PDF here after generation
  # Intelligence Assessment coverage is governed by triage.max_main_items

storage:
  db_path: "./data/briefing.db"
  media_dir: "./data/media"
  retention_days: 30          # delete media older than this many days
  # analysis_lookback_days: 30  # how far back --batch reaches for unanalysed posts, and
                                # the cap on lookback_seconds. Defaults to retention_days
```

`analysis_lookback_days` is the **analysis window**. Past `retention_days` the media a
post references has already been purged, so analysing it means analysing it blind on its
text — by default `--batch` therefore reaches back exactly as far as media is kept. Older
unanalysed posts stay queued and are reachable with an explicit `--since`, which overrides
the window in both directions. Setting the field higher than `retention_days` is allowed
but only buys text-only analyses of old posts.

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

If LM Studio is on a different machine, set `server_host` in `config.yaml` to its IP address (e.g. `192.168.1.96`). Note that the connection is plain HTTP — the API token and post content traverse the LAN unencrypted, so only do this on a trusted network.

### Step 4 — (Optional) Entity aliases for better deduplication and trends

Dedup and 7-day mention trends compare named entities after lowercasing and stripping periods, so `"U.S."` and `"US"` already match. For naming variants that need an explicit mapping (acronyms, alternate spellings, regional name variants), copy the example and add your own:

```bash
cp entity_aliases.yaml.example entity_aliases.yaml
```

```yaml
# entity_aliases.yaml
u.s.: united states
dprk: north korea
```

This file is empty by default and gitignored — there are no built-in aliases, so without it entity normalisation is a no-op beyond the lowercase/strip-periods rule. Tailor it to the actors and regions your configured channels actually cover.

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
3. Re-downloads media for any post inside the analysis window whose image file is missing — either the original download failed, or `purge_old_media()` has since deleted it. Capped at 200 posts per run; a post whose message Telegram no longer serves is left alone and analysed as text-only
4. Sends each new post to LM Studio for analysis — headline title, importance, urgency, credibility, relevance, category, threat level, and key entities. Posts with fewer than 30 characters of text and no media are skipped without an LLM call
5. Runs triage: scores posts (channel priority × credibility, keyword boost, rumour penalty, recency decay), clusters cross-channel reports of the same story — the best report is kept and the rest become "corroborated by" references that boost its score — then splits into main/appendix
6. Generates `briefings/YYYY-MM-DD/TheDailyTelegram_YYYY-MM-DD_HHMMSS.pdf`
7. Sends the triaged main items, a 7-day entity/category mention-trend table, and yesterday's assessment themes to LM Studio for intelligence synthesis
8. Prepends a structured intelligence front page (situation summary, key themes with source citations and continuity tags, signals & warnings, named actors, emerging actors) to the briefing PDF, and persists the assessment for next-day continuity
9. Disconnects

Subsequent `--batch` runs on the same day are safe — cursor tracking ensures no post is fetched twice, and UNIQUE constraints prevent duplicate DB entries. If LM Studio is unreachable during the front page step, a warning is logged and the briefing PDF is kept as-is.

### The analysis window

`--batch` analyses unanalysed posts from the last `storage.analysis_lookback_days` days only, which defaults to `storage.retention_days` (30). Older posts stay in the queue untouched.

The reason is media. `purge_old_media()` deletes image files older than `retention_days`, but the post row keeps referencing them, so analysing a post past that point means analysing it blind on its text while the briefing still renders an "Image" line. Reaching back no further than media is kept keeps the two in step.

The window also caps `telegram.lookback_seconds`, so resetting cursors can't pull in months of history whose images will never exist.

One consequence worth knowing: posts older than the window accumulate in the database as permanently unanalysed. They are counted in a log line on each run and are not lost — an explicit `--since` reaches them — but the unanalysed count is no longer a useful health signal on its own.

```
INFO Analysis window: 16743 older unanalysed posts excluded from this run (predate 2026-08-06T17:02:15+00:00), still queued for a later run with an explicit --since
```

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

**`--since` overrides the analysis window in both directions.** It replaces the derived cutoff outright, so `--since 2026-06-01` reaches back past `analysis_lookback_days` and `--since 08:00` narrows the run to today. It also overrides the cap on `lookback_seconds`, since it is an explicit statement of intent.

> **Use `--since` only when you intentionally need a historical lookback.** For routine same-day re-runs, use plain `--batch` — it uses the cursor and fetches only messages that arrived since the last run.

---

## Daemon mode — live monitoring

Daemon mode runs indefinitely, listening for new messages in real time and generating a PDF automatically at the configured `generate_at` time each day.

**How posts get analysed.** The daemon stores each post the moment it arrives,
then analyses newly stored posts in a sweep every 10 minutes
(`DAEMON_ANALYSIS_INTERVAL_SECS` in `main.py`) rather than one at a time on
arrival. Analysis is therefore *eventually* consistent — a post is analysed
within a sweep interval, not instantly — which is invisible to a once-nightly
briefing and is what lets the daemon use a slower, better model (see
[Run modes and analysis profiles](#run-modes-and-analysis-profiles)). Posts that
arrive mid-sweep are picked up by the next one; the queue lives in the database,
so nothing is lost if the daemon restarts.

The sweep only considers posts stored **since the daemon started**. It will not
work through a historical backlog — that remains `--batch`'s job.

> **Important:** The daemon is a live listener only. It processes messages that arrive while it is running — it does **not** backfill historical posts. Always run `--batch` first to catch up on any posts you want in the briefing, then start the daemon.

> **Warning:** Do not run `--batch` (or `--generate`/`--analyse`) while the daemon is running. Both share one Telegram session file and one SQLite database — running them concurrently can lock the session or the database. Stop the daemon first (`Ctrl+C` or `kill`), run the catch-up command, then restart the daemon.

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
2. Inserts a `PostRecord` into SQLite — analysis happens later, in the periodic sweep described above, not inline
3. Advances the channel's cursor, so a later `--batch` resumes from here
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

> **Note:** `--generate` always builds today's (UTC) briefing — it does not accept `--since` and cannot rebuild a past day's PDF from scratch. `--analyse --since <date>` only prepends the intelligence front page to a PDF that already exists for that date; it does not regenerate the briefing itself.

### Mobile layout — `--layout`

By default the PDF uses the desktop CSS layout. Pass `--layout mobile` (with `--batch`, `--generate`, or `--daemon`) to use a layout with larger text, tighter margins, and a single-column appendix, optimised for reading on a phone:

```bash
python -m tg_compiler.main --generate --layout mobile
```

The default can also be set permanently via `generation.pdf_layout` in `config.yaml` (`"desktop"` or `"mobile"`); `--layout` overrides it for a single run.

### Sharing the PDF — `share_to_directory`

Set `generation.share_to_directory` in `config.yaml` to a directory path, and the final generated PDF (after the intelligence front page has been prepended) is copied there too — e.g. a Syncthing/Dropbox/Nextcloud folder for reading on another device. Leave it unset (the default) to disable this.

---

## Run modes and analysis profiles

The two ways of analysing posts have opposite constraints:

- **`--batch` / `--since`** processes a whole day at once. A thousand-plus posts
  have to be analysed in one sitting, so speed decides whether the run finishes.
- **`--daemon`** watches channels live. Posts trickle in, there is idle time
  between them, and a slower but better model can be afforded.

An **analysis profile** is a named set of LM Studio settings for one mode.
`--batch`/`--since` select the `batch` profile, `--daemon` selects `daemon`, and
`--analysis-profile NAME` overrides either:

```bash
python -m tg_compiler.main --batch                          # uses the "batch" profile
python -m tg_compiler.main --daemon                         # uses the "daemon" profile
python -m tg_compiler.main --daemon --analysis-profile batch  # daemon, but with the fast model
```

A profile carries **the model and the settings that must accompany it** — token
budgets, concurrency, context length, batch sizes — because those are
model-specific. A reasoning model needs a large `analysis_base_tokens` or its
JSON is truncated mid-deliberation; a small model needs a fraction of that, and
several large-budget image requests at once will exhaust LM Studio's context.
Carrying only a model name between modes reliably breaks one of them.

```yaml
lmstudio:
  analysis_profiles:
    batch:
      model: "mistralai/ministral-3-3b"
      analysis_base_tokens: 700
      analysis_max_tokens: 1600
      max_concurrent_analyses: 4
      model_context_length: 32768
    daemon:
      model: "prism-ml/bonsai-27b"
      analysis_base_tokens: 9500
      analysis_max_tokens: 16000
      max_concurrent_analyses: 1
      model_context_length: 32768
      batch_size: 10
      batch_size_with_images: 3
```

Omit `analysis_profiles` entirely and both modes use the plain `lmstudio`
settings, exactly as before. A profile only states what differs; everything else
is inherited. Naming a profile that doesn't exist with `--analysis-profile` is an
error; a *mode default* that doesn't exist simply falls back to the plain
settings.

**Pick a daemon model equal to `synthesis_model` where you can.** If they differ
and `manage_models` is on, every analysis sweep and every nightly generation
swaps weights (~16 s per load), which at a 10-minute cadence is pure waste.

---

## Throughput — choosing an analysis model, batch size and concurrency

Analysis dominates runtime: it touches every scraped post (1,000–1,700 a day on a
16-channel setup), while everything downstream runs once. The settings below were
chosen from measurements on an RTX 3080 Ti Laptop (16 GB), and the numbers are
recorded here so they can be re-checked rather than guessed at.

**Reasoning tokens are the thing that matters.** On a reasoning model, 93–97% of
every generated token is deliberation the briefing never sees:

| model | s/post (one post per call) | reasoning tokens/post |
|---|---|---|
| `prism-ml/bonsai-27b` (reasoning) | 85.8 | 3,878 |
| `mistralai/ministral-3-3b` | 2.2–3.0 | **0** |
| `google/gemma-3-4b` | 2.9 | 0 |

LM Studio exposes no way to cap reasoning separately — `reasoning_effort`,
`chat_template_kwargs.enable_thinking` and `reasoning: {effort}` were all accepted
and ignored — so the only remedy is a model that does not deliberate.

**Batching helps reasoning models, and only them.** Several posts in one call
amortise the per-call deliberation. On `bonsai-27b` a batch of 10 cut reasoning
from 3,515 to 693 tokens per post (191 → 39.6 s/post). On a non-reasoning model
there is nothing to amortise, and batching costs output quality:

| batch (ministral) | s/post | mean summary chars | mean entities |
|---|---|---|---|
| **1** | 3.0 | **281** | **3.6** |
| 5 | 2.0 | 180 | 2.8 |
| 10 | 1.8 | 179 | 2.3 |

Batch 10 buys 1.7x on text (1.07x on images) for 36% shorter summaries, 36% fewer
entities, and only ~50% category agreement with the model's own single-post
output. **So: keep `batch_size: 1` for a fast model; raise it only if you point
analysis at a reasoning model.** Everything the batch path adds — index mapping,
the anchor check, salvage, the fallback ladder — is inert at size 1.

**Concurrency is the better lever for a small model**, but its ceiling is set by
total concurrent context, not by the request count:

| configuration | s/post | outcome |
|---|---|---|
| batch 1, `max_concurrent_analyses: 1` | 2.17 | all OK |
| batch 1, `max_concurrent_analyses: 4` | **1.10** | all OK |
| batch 10, `max_concurrent_analyses: 4` | — | **every request failed** |

Large concurrent requests exhaust context and fail outright. Keep
`analysis_*` token budgets sized for the model actually doing the analysis: a
budget tuned for a reasoning model (`analysis_max_tokens: 16000`) reserves far
more than a small model needs, and four such image-post requests at once will
fail. A non-reasoning model needs roughly 250 output tokens per post.

Realistic end-to-end figure for a full day of 1,239 real posts with
`batch_size: 1` and `max_concurrent_analyses: 4`: **2.6 s/post, ~52 minutes**.
Microbenchmarks on short text-only posts report ~1.1 s/post; a real day includes
image posts and long posts, so expect the higher figure.

**A large model cannot keep up post-by-post.** This is why the daemon batches
rather than analysing each post on arrival:

| | 1,100 posts/day | 1,700 posts/day |
|---|---|---|
| bonsai-27b, one post per call (85.8 s/post) | 26 h/day | 40 h/day |
| bonsai-27b, **batched ×10** (23.1 s/post) | **7 h/day** | 11 h/day |

Channels here produce 750–1,700 posts a day. One at a time, a reasoning model
needs more hours per day than a day contains and falls permanently behind;
batched, it fits. So a `daemon` profile using a reasoning model should set
`batch_size: 10` — the one case where batching earns its cost.

**Model swapping.** With `manage_models: true` the analyzer and synthesiser each
load their own model at stage entry via the LM Studio Python SDK, unloading the
other first. This is required when the two models cannot co-reside in VRAM — a
27B model occupies ~12 GB of a 16 GB card. With `manage_models: false` (the
default) nothing is loaded or unloaded and whatever is already running is used.

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
- **Threat level badge**: 🟥 CRITICAL · 🟧 HIGH · 🟨 MODERATE · 🟩 LOW
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

In the PDF these are coloured square glyphs (`#c0392b`, `#d35400`, `#b7950b`, `#1e8449`).

| Badge | Level | Meaning |
|---|---|---|
| 🟥 | CRITICAL | Imminent risk of mass casualties, confirmed state-level military action underway, nuclear/chemical/biological threat, or active attack on critical infrastructure |
| 🟧 | HIGH | Confirmed armed conflict development, significant political crisis, major terror attack, or credible escalation warning from a named senior state official |
| 🟨 | MODERATE | Ongoing conflict updates, diplomatic developments, significant arrests or detentions, or unverified but plausible escalation claims |
| 🟩 | LOW | Background context, routine troop movement reports, unverified rumours, social media content, statistical or historical reports |

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
Telegram rate-limited the request. Waits of 600 seconds or less are handled automatically — the scraper pauses and resumes on its own. Longer waits abort that channel's scrape with partial results saved; it will pick up where it left off on the next `--batch` run. If it happens often, increase `rate_limit_delay_ms`.

**"ValidationError: extra inputs are not permitted"**  
A field in `config.yaml` is misspelled or unknown. Check the field name against `config.yaml.example`.

**"TG_API_ID env var must be an integer"**  
The `TG_API_ID` environment variable is set but contains a non-integer value. Either unset it or fix the value.

**PDF is empty or has no posts**  
Either no posts were scraped today, or LM Studio analysis has not run yet. Run `--batch` to trigger a full scrape+analyse cycle, then check `--generate`.

**Intelligence front page not prepended**  
If LM Studio was unreachable during the synthesis step, a warning is logged and the briefing is kept as-is. Check LM Studio is running and retry with `--analyse`.

**"Could not read image … No such file or directory" while analysing**  
Should no longer appear. The path exists in the database but the file was purged by `retention_days` or never downloaded; the analyzer now filters absent paths before building the prompt, and `--batch` re-downloads what it can first. If you do see it, the file exists but cannot be decoded (a truncated download) — it is logged at DEBUG and the post is analysed as text-only.

**The unanalysed post count keeps growing**  
Expected. Posts older than `storage.analysis_lookback_days` are skipped by routine runs and stay queued; each run logs how many. To analyse them anyway, run `--batch --since <date>` — but note their media is already gone, so the result is text-only.

**Session file issues after moving the project**  
Delete `<session_name>.session` and re-authenticate by running `--batch` again.

**Duplicate posts from a channel across old batch and daemon runs**  
Older versions stored `channel_id` inconsistently between batch and daemon runs, which could let the same message slip past the `UNIQUE(channel_id, message_id)` constraint under two different IDs. `scripts/migrate_channel_ids.py` is a one-off migration that backs up the DB, merges the duplicate rows, and rebuilds `channel_cursors` under a consistent ID. Run it once against an affected database:
```bash
python scripts/migrate_channel_ids.py ./data/briefing.db
# or preview without writing:
python scripts/migrate_channel_ids.py ./data/briefing.db --dry-run
```
Not needed on a fresh database — only for one created before this fix.

---

## Running tests

```bash
source .venv/bin/activate
pytest                          # all tests
pytest tests/test_db.py -v      # single file
pytest tests/test_triage.py::test_composite_score_formula -v   # single test
```

Tests use in-memory SQLite and do not require Telegram credentials or a running LM Studio server.

---

## Inspecting the database

The pipeline's own output — the PDF — is already triaged and filtered. To look at what the
analysis stage actually produced, browse `data/briefing.db` directly with
[Datasette](https://datasette.io/), an optional dev dependency:

```bash
source .venv/bin/activate
pip install -e ".[inspect]"                                    # one-off, ~15 pure-Python packages

datasette data/briefing.db \
  -m scripts/datasette_metadata.yaml \
  --plugins-dir scripts/datasette_plugins
# then open http://localhost:8001
```

WSL2 forwards `localhost`, so a browser on the Windows side reaches it with no extra flags.

**This cannot corrupt or modify the database.** Datasette opens the file through a
`file:...?mode=ro` SQLite URI and additionally rejects any statement that is not a `SELECT`,
so it is safe to browse while `--batch` or `--daemon` is writing — WAL readers and the
writer do not block each other, and you see committed data live.

> **Never pass `-i` / `--immutable` against `data/briefing.db`.** That flag asserts to SQLite
> that the file will never change, which is false during a run, and yields incorrect reads
> rather than a clean error. It is only legitimate against a snapshot (see below).

`scripts/datasette_metadata.yaml` sets up faceted browsing (channel on `posts`; category,
threat level and model on `analyses`) plus these canned queries, linked from the database page:

| Query | What it answers |
|---|---|
| `analysed_posts` | The denormalised analysis-with-source-post join. Filter by channel, date and threat level; blank fields are ignored. |
| `threat_by_channel` | Threat-level mix and mean scores per channel. |
| `category_counts` | Category distribution, including how much the content gate marks `Skipped`. |
| `score_distribution` | Histogram across all four scoring axes. |
| `model_comparison` | Output richness per `model_used` — summary length, entity count, image-insight rate — over production output rather than a benchmark sample. |
| `unanalysed_backlog` | Posts with no `analyses` row, by day and channel. |
| `recent_leads` | CRITICAL / HIGH items from the last N days (default 7). |
| `intel_history` | The stored per-day synthesised assessments. |

The configuration adds no views, tables or indexes — nothing for `init_schema()` to collide with.

Four plugins ship with the `inspect` extra and load automatically:

| Plugin | What it adds |
|---|---|
| `datasette-media` | Serves the scraped images at `/-/media/photo/<post id>`. |
| `datasette-json-html` | Turns a JSON cell into real HTML — inline thumbnails and clickable links. |
| `datasette-pretty-json` | Formats the four JSON columns (`key_entities`, `media_paths`, `raw_json`, `intel_json`). |
| `datasette-vega` | Point-and-click charts on any query result, from dropdowns above the table. |

The payoff is the **`image_review`** query: a thumbnail of each scraped image beside the
model's own `image_insights` for it, so image-analysis quality can be judged at a glance.
4,194 analysed posts qualify; the query shows the 50 most recent, because images are served
at full size (~143 KB each) rather than resized.

Two things worth knowing:

- **Run Datasette from the project root.** `media_paths` stores paths relative to it
  (`data/media/...`), and `datasette-media` opens them as given.
- **`posts.channel_name` is the slug, not the Telegram username** — they differ for 7 of the
  16 channels (`WarFrontWitness` → `wfwitness`, `RerumNovarum` → `rnintel`, …). Building a
  `t.me/<channel_name>/...` link in SQL therefore produces dead links. Use the
  `tme_link(slug, message_id)` SQL function instead: `scripts/datasette_plugins/channel_links.py`
  registers it, backed by `AppConfig.channel_link_map()` so it cannot drift from `config.yaml`.
  It returns `NULL` for a channel with no configured username, and the viewer still starts if
  `config.yaml` is missing.

`datasette-vega` is unmaintained (last release 2018) but bundles vega-lite offline and only
injects static assets, so it works fine here. If a future Datasette upgrade ever breaks it,
drop it from the `inspect` extra — nothing else depends on it.

### Stopping it — the "Stop server" button

`scripts/datasette_plugins/shutdown_button.py` adds a red **Stop server** button to the
bottom-right of every page. It asks for confirmation, then shuts Datasette down gracefully:
uvicorn finishes in-flight requests, closes connections, and the `datasette` command exits 0,
returning your terminal to a prompt. `Ctrl-C` in the terminal does the same thing.

The database is never at risk — Datasette holds only `mode=ro` connections, so there is
nothing to flush or roll back. `pragma integrity_check` returns `ok` afterwards.

The endpoint that does this (`POST /-/shutdown`) stops a process, so it is guarded three ways:
POST only, CSRF-checked by Datasette's own middleware, and refused unless the request's `Host`
is loopback — so the button is inert if you ever bind to a public interface with `-h 0.0.0.0`.

**On closing the browser tab:** the button tries, but browsers only permit a page to close
itself when a script opened that window in the first place. A tab you opened by typing the URL
will not close — Chrome and Firefox both block it. So instead of silently failing, the page
replaces itself with a "Datasette has stopped" panel telling you the tab is now safe to close
with `Ctrl+W`. If you want genuine one-click closing, launch the UI from a script-opened
window; there is no way to get it from a normally-opened tab.

For heavy exploratory queries, work from a snapshot instead of the live file. `.backup` is an
online, WAL-safe copy, and `--immutable` is safe (and faster) on the copy:

```bash
sqlite3 data/briefing.db ".backup /tmp/briefing-snap.db"
datasette -i /tmp/briefing-snap.db -m scripts/datasette_metadata.yaml
```

A cold first run of `unanalysed_backlog` scans all posts and can brush against Datasette's
default 1-second query limit; add `--setting sql_time_limit_ms 8000` if you hit it.

For a quick one-off without the UI, the same read-only guarantee applies to the CLI:

```bash
sqlite3 -box "file:data/briefing.db?mode=ro" "select count(*) from analyses;"
```

---

## Project layout

For a module-by-module breakdown of the pipeline (`scraper.py`, `analyzer.py`, `triage.py`, `generator.py`, `synthesiser.py`, `trends.py`, `db.py`, etc.) and the data-flow contracts between them, see [`CLAUDE.md`](./CLAUDE.md).
