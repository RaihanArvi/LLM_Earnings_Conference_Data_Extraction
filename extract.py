from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import DefaultAioHttpClient
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
import textwrap
import asyncio
import logging
from tqdm import tqdm

INPUT_FILE = "transcripts_dataset/2021_q4_clean.csv"
SEM = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

time_now = datetime.now()

client = AsyncOpenAI(
        http_client=DefaultAioHttpClient())

def async_retry():
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(min=1, max=10),
        reraise=True,
    )


# ----------------------------
# Concurrency control
# ----------------------------
sem = asyncio.Semaphore(SEM)


# ----------------------------
# Models
# ----------------------------

class ExtractResponse(BaseModel):
    representative_quotes: List[str]
    quantitative_summary: List[str]
    qualitative_summary: List[str]

# Data or SDK

# class DocumentSummary(BaseModel):
#     firm_name: Optional[str] = Field(
#         default=None, description="Company name or null if unknown"
#     )
#     call_date: Optional[str] = Field(
#         default=None, description="Earnings call date or null"
#     )
#     source_type: str = Field(
#         default="earnings_call",
#         description="Source type, fixed to 'earnings_call'"
#     )

class Excerpt(BaseModel):
    # excerpt_id: int = Field(
    #     ..., description="Sequential excerpt identifier"
    # )
    paragraph_text: str = Field(
        ..., description="Verbatim paragraph from the transcript"
    )
    raw_keywords_detected: List[str] = Field(
        ...,
        description="List of detected key terms or vendor names"
    )
    labels: List[str] = Field(
        ...,
        description="Classification labels applied to the excerpt"
    )
    notes: str = Field(
        default="",
        description="Optional short summary of overall relevance"
    )

# class EarningsCallExtraction(BaseModel):
#     document_summary: DocumentSummary
#     excerpts: List[Excerpt]

class LLMAnnotation(BaseModel):
    is_data_or_ai: Optional[int] = Field(
        default=None,
        description="Indicates whether the content relates to data or AI"
    )
    is_sdk_or_data: Optional[int] = Field(
        default=None,
        description="Indicates whether the content relates to data or AI"
    )
    is_att: Optional[int] = Field(
        default=None,
        description="Indicates whether the content relates to data or AI"
    )
    representative_quotes: Optional[str] = Field(
        default=None,
        description="Representative verbatim quotes supporting the classification"
    )
    quantitative_summary: Optional[str] = Field(
        default=None,
        description="Concise quantitative summary, if applicable"
    )
    qualitative_summary: Optional[str] = Field(
        default=None,
        description="Concise qualitative summary, if applicable"
    )
    excerpt_data_or_sdk_extraction: Optional[Excerpt] = Field(
        default=None,
        description="Extracted excerpt from the transcript"
    )
    excerpt_att_extraction: Optional[Excerpt] = Field(
        default=None,
        description="Extracted excerpt from the transcript"
    )


# ----------------------------
# Classifiers
# ----------------------------
@async_retry()
async def detect_data_or_ai(passage: str) -> int:
    prompt = f"""You are a classifier. Read the following passage and determine if the company talks
about topics related to 'data' or 'AI' (artificial intelligence), including concepts
like analytics, machine learning, algorithms, data-driven decision making, big data,
or AI applications.

Respond with ONLY a single number:
- 1 if the passage discusses AI or data in any meaningful way.
- 0 if it does not.

Passage:
\"\"\"{passage}\"\"\"
    """

    response = await client.responses.create(
        model="gpt-5-mini",
        instructions="You are a strict binary classifier.",
        reasoning={"effort": "medium"},
        text={"verbosity": "high"},
        input=[
            {"role": "user", "content": prompt}
        ]
    )

    try:
        result = int(response.output_text)
        return 1 if result == 1 else 0
    except:
        return 0

@async_retry()
async def detect_sdk_or_data(passage: str) -> int:
    prompt = f"""You are a classifier. Read the following passage and determine the passage is relevant. Refer to the guidelines below for relevancy.

## What counts as “relevant” (keywords and concepts)

Treat a paragraph as relevant if it clearly discusses any of the topics below. Use both exact
matches and close paraphrases.

---

### SDK / Data Infrastructure Concepts

Include both generic concepts and specific vendor/SDK names.

#### Generic terms (exact or close variants):
- “SDK”, “software development kit”
- “mobile analytics”, “app analytics”
- “mobile attribution”, “ad attribution”
- “measurement partner”, “measurement”, “measurement stack”
- “third-party analytics”, “third-party measurement”
- “data signals”, “attribution signals”, “signal loss”
- “app engagement data”, “user engagement data”, “in-app events”
- “data infrastructure”, “data pipeline”, “measurement stack”, “data stack”

#### Specific SDKs / vendors (brand names):
- Firebase, “Google Analytics for Firebase”, Google Analytics
- Facebook SDK, Meta Audience Network
- Adobe Analytics, Adobe Experience Cloud
- Appsflyer, Adjust, Kochava, Branch
- Unity Ads, ironSource, AppLovin, MoPub, Segment, Flurry, Amplitude, Mixpanel

Mark a paragraph as relevant if it describes:
- installing, using, or changing SDKs,
- reliance on mobile analytics / attribution tools,
- measurement partners in mobile advertising,
- the “measurement stack” or data infrastructure for ads / app performance.

---

Respond with ONLY a single number:
- 1 if the passage is relevant.
- 0 if it is not relevant.

Passage:
\"\"\"{passage}\"\"\"
    """

    response = await client.responses.create(
        model="gpt-5-mini",
        instructions="You are a strict binary classifier.",
        reasoning={"effort": "medium"},
        text={"verbosity": "high"},
        input=[
            {"role": "user", "content": prompt}
        ]
    )

    try:
        result = int(response.output_text)
        return 1 if result == 1 else 0
    except:
        return 0


@async_retry()
async def detect_att(passage: str) -> int:
    prompt = f"""You are a classifier. Read the following passage and determine the passage is relevant. Refer to the guidelines below for relevancy.

## What counts as “relevant” (keywords and concepts)

Treat a paragraph as relevant if it clearly discusses any of the topics below. Use both exact
matches and close paraphrases.

---

### ATT / IDFA / iOS Privacy Shock Concepts

Look for explicit or implicit references to Apple’s privacy changes.

#### Terms:
- “App Tracking Transparency”, “ATT”
- “IDFA”, “IDFA changes”, “limited access to IDFA”
- “iOS 14”, “iOS 14.5”
- “privacy changes”, “privacy update”, “Apple privacy policy”

#### Effects on measurement/targeting:
- “measurement disruption”, “lost signals”, “signal loss”
- “reduced accuracy”, “less precise targeting”, “targeting limitations”
- “harder to attribute campaigns”, “harder to measure ROAS”

#### Effects on data:
- “limited tracking on iOS”
- “less visibility on user behavior”
- “fewer attribution signals”

Mark a paragraph as relevant if:
- ATT / IDFA / iOS privacy is mentioned directly, or
- the company describes Apple privacy changes clearly causing:
  - signal loss, degraded measurement, targeting/optimization issues, or
  - financial impact specifically tied to those changes.

Respond with ONLY a single number:
- 1 if the passage is relevant.
- 0 if it is not relevant.

Passage:
\"\"\"{passage}\"\"\"
    """

    response = await client.responses.create(
        model="gpt-5-mini",
        instructions="You are a strict binary classifier.",
        reasoning={"effort": "medium"},
        text={"verbosity": "high"},
        input=[
            {"role": "user", "content": prompt}
        ]
    )

    try:
        result = int(response.output_text)
        return 1 if result == 1 else 0
    except:
        return 0


# ----------------------------
# Extractors
# ----------------------------

@async_retry()
async def extract(passage: str) -> ExtractResponse:
    """Return representative quotes and summaries for passages about data/AI."""
    analyst_instructions = """You are an analyst focused on how companies discuss data and AI in earnings calls. Follow the provided formatting rules exactly."""

    user_prompt = textwrap.dedent(
        f"""Goal:
Identify how this earnings call excerpts talks about 'data' and 'AI'. Focus on
meaningful mentions tied to strategy, products, operations, risks, or investments.

Format:
1. Representative Quotes (short excerpts that explicitly mention 'data' or 'AI', make sure it's a direct quote and do not rephrase it).
2. Quantitative Summary: A list of bullet points (written as strings) covering non-exhaustive quantitative aspects such as:
   - Numbers, metrics, percentages, growth rates
   - Financial figures, revenue, costs, investments
   - Performance metrics, KPIs, benchmarks
   - Any measurable or numerical data related to data/AI
3. Qualitative Summary: A list of bullet points (as strings) covering non-exhaustive qualitative aspects such as:
   - Strategic direction, vision, goals
   - Product features, capabilities, innovations
   - Market positioning, competitive advantages
   - Risks, challenges, opportunities
   - Any non-numerical insights related to data/AI

Important:
- Both 'Quantitative Summary' and 'Qualitative Summary' have to be related to 'data' or 'AI'.
- If no quantitative or qualitative insights are found, return empty string.

Each summary should be a list of concise bullet point strings (3-8 bullet points each).

Passage:
\"\"\"{passage}\"\"\"""").strip()

    try:
        response = await client.responses.parse(
            model="gpt-5-mini",
            instructions=analyst_instructions,
            reasoning={"effort": "medium"},
            text={"verbosity": "high"},
            input=[
                {"role": "user", "content": user_prompt},
            ],
            text_format=ExtractResponse,
        )
        return response.output_parsed

    except Exception:
        return ExtractResponse(representative_quotes=[], quantitative_summary=[], qualitative_summary=[])

@async_retry()
async def extract_data_or_sdk(passage: str) -> Excerpt:
    """Return representative quotes and summaries for passages about data/AI."""
    analyst_instructions = """You are analyzing transcripts (e.g., earnings calls, investor presentations, interviews) to identify and classify paragraphs related to mobile SDKs, data infrastructure, third-party measurement.

Your task is to:
- identify relevant paragraphs using the relevance criteria below,
- extract those paragraphs in full,
- assign appropriate classification labels (multi-label allowed),
- and capture structured metadata when available.

Be precise and consistent. Do not infer information that is not explicitly or clearly implied in the text."""

    user_prompt = textwrap.dedent(
        f"""For each extracted paragraph, assign one or more of the following labels (multi-label is allowed):

Use these exact words for label: ["SDK/Data Infrastructure Reliance", "Third-Party Data Dependency"]

### 1. SDK/Data Infrastructure Reliance
Use when the paragraph discusses:
- SDKs, mobile analytics, attribution tools, or ad tech vendors;
- measurement partners, analytics partners;
- the data/measurement “stack” or infrastructure used to run and optimize campaigns, track users, or measure engagement.

### 2. Third-Party Data Dependency
Use when the paragraph emphasizes dependence on:
- external partners, measurement vendors, ad networks, DSPs;
- “third-party data signals,” “third-party measurement,” “partner APIs,” external data feeds, or aggregated data provided by partners.

If a paragraph is clearly relevant but doesn’t fit any of these, you may assign no label (empty list), but that should be rare. Prefer to use all applicable labels.

---

## 3. How to segment and capture metadata

- Extract full paragraphs from the transcript (not truncated sentences).
- If some metadata is not available in the text, set it to null.

Passage:
\"\"\"{passage}\"\"\"""").strip()

    try:
        response = await client.responses.parse(
            model="gpt-5-mini",
            instructions=analyst_instructions,
            reasoning={"effort": "medium"},
            text={"verbosity": "high"},
            input=[
                {"role": "user", "content": user_prompt},
            ],
            text_format=Excerpt,
        )
        return response.output_parsed

    except Exception:
        return Excerpt(paragraph_text="", speaker="", section=None, timestamp=None, raw_keywords_detected=[], labels=[], notes="")


@async_retry()
async def extract_att(passage: str) -> Excerpt:
    """Return representative quotes and summaries for passages about data/AI."""
    analyst_instructions = """You are analyzing transcripts (e.g., earnings calls, investor presentations, interviews) to identify and classify paragraphs related to Apple’s iOS privacy changes (ATT/IDFA).

Your task is to:
- identify relevant paragraphs using the relevance criteria below,
- extract those paragraphs in full,
- assign appropriate classification labels (multi-label allowed),
- and capture structured metadata when available.

Be precise and consistent. Do not infer information that is not explicitly or clearly implied in the text."""

    user_prompt = textwrap.dedent(
        f"""For each extracted paragraph, assign one or more of the following labels (multi-label is allowed):

Use these exact words for label: ["ATT Impact – Direct (Signal/Measurement)", "ATT Impact – Financial"]

### 1. ATT Impact – Direct (Signal/Measurement)
Use when the paragraph links ATT/IDFA/iOS privacy changes to:
- reduced accuracy of measurement or targeting,
- “signal loss,” fewer attribution signals,
- inability to track or attribute users/campaigns,
- degraded optimization, difficulties in campaign performance measurement.

This label focuses on information / measurement / optimization consequences.

### 2. ATT Impact – Financial
Use when the paragraph links ATT/IDFA/iOS privacy changes to financial outcomes, e.g.:
- lower ad revenue, weaker ROAS, higher customer acquisition cost,
- missed guidance or forecast revisions,
- explicit revenue or profit shortfalls attributed to privacy changes.

This label focuses on earnings, revenue, margins, or guidance effects.

If a paragraph is clearly relevant but doesn’t fit any of these, you may assign no label (empty list), but that should be rare. Prefer to use all applicable labels.

---

## 3. How to segment and capture metadata

- Extract full paragraphs from the transcript (not truncated sentences).
- If some metadata is not available in the text, set it to null. 

Passage:
\"\"\"{passage}\"\"\"""").strip()

    try:
        response = await client.responses.parse(
            model="gpt-5-mini",
            instructions=analyst_instructions,
            reasoning={"effort": "medium"},
            text={"verbosity": "high"},
            input=[
                {"role": "user", "content": user_prompt},
            ],
            text_format=Excerpt,
        )
        return response.output_parsed

    except Exception:
        return Excerpt(paragraph_text="", speaker="", section=None, timestamp=None, raw_keywords_detected=[], labels=[], notes="")

# ----------------------------
# Helper
# ----------------------------

def format_quotes(quotes: List[str]) -> str:
    """Return quotes formatted in "","" style."""
    cleaned = []
    for quote in quotes or []:
        cleaned_quote = " ".join(quote.split())
        if cleaned_quote:
            cleaned.append(cleaned_quote)
    if not cleaned:
        return ""
    return '"' + '","'.join(cleaned) + '"'


# ----------------------------
# Process Row
# ----------------------------

async def process_row(row: pd.Series) -> LLMAnnotation:
    """Run classification and extraction (if appropriate) for a single dataframe row."""
    annotation = LLMAnnotation()
    passage = str(row.get("componenttext", "") or "")

    #classification = detect_data_or_ai(passage)
    # if classification == 1:
    #     annotation.is_data_or_ai
    #     extraction = extract(passage)
    #     formatted_quotes = format_quotes(extraction.representative_quotes)
    #     formatted_quantitative = format_quotes(extraction.quantitative_summary)
    #     formatted_qualitative = format_quotes(extraction.qualitative_summary)

    data_or_sdk_relevancy = await detect_sdk_or_data(passage)
    annotation.is_sdk_or_data = data_or_sdk_relevancy
    if data_or_sdk_relevancy == 1:
        excerpt_data_or_sdk_extraction = await extract_data_or_sdk(passage)
        annotation.excerpt_data_or_sdk_extraction = excerpt_data_or_sdk_extraction

    att_relevancy = await detect_att(passage)
    annotation.is_att = att_relevancy
    if att_relevancy == 1:
        excerpt_att_extraction = await extract_att(passage)
        annotation.excerpt_att_extraction = excerpt_att_extraction

    return annotation

async def sem_process_row(row):
    async with sem:
        return await process_row(row)

# ----------------------------
# Run all rows
# ----------------------------

async def run_all(df):
    async def wrapped(idx, row):
        try:
            annotation = await sem_process_row(row)
            return idx, annotation
        except Exception:
            logging.exception("Error processing index %s", idx)
            raise

    tasks = [
        asyncio.create_task(wrapped(idx, row))
        for idx, row in df.iterrows()
    ]

    completed_count = 0
    for completed in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        idx, annotation = await completed
        completed_count += 1

        logging.info("Completed processing index %s (%s/%s)", idx, completed_count, len(tasks))

        # df.at[idx, "is_data_or_ai"] = annotation.is_data_or_ai
        # df.at[idx, "representative_quotes"] = annotation.representative_quotes
        # df.at[idx, "quantitative_summary"] = annotation.quantitative_summary
        # df.at[idx, "qualitative_summary"] = annotation.qualitative_summary

        df.at[idx, "is_sdk_or_data"] = annotation.is_sdk_or_data
        if annotation.is_sdk_or_data != 0:
            df.at[idx, "excerpt_data_or_sdk_extraction_json"] = (
                annotation.excerpt_data_or_sdk_extraction.model_dump_json()
            )
        else:
            df.at[idx, "excerpt_data_or_sdk_extraction_json"] = None

        df.at[idx, "is_att"] = annotation.is_att
        if annotation.is_att != 0:
            df.at[idx, "excerpt_att_extraction_json"] = (
                annotation.excerpt_att_extraction.model_dump_json()
            )
        else:
            df.at[idx, "excerpt_att_extraction_json"] = None

        if completed_count % 500 == 0:
            output_filename = time_now.strftime("processed_transcripts_%Y%m%d_%H%M%S.csv")
            output_path = f"output/{output_filename}"
            df.to_csv(output_path, index=False)

    # Final Save
    final_filename = time_now.strftime("processed_transcripts_FINAL_%Y%m%d_%H%M%S.csv")
    final_path = f"output/{final_filename}"
    df.to_csv(final_path, index=False)
    logging.info("Final file saved to %s", final_path)

    await client.close()


# ----------------------------
# Load Dataframe
# ----------------------------
def load_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.iloc[0:100]

    processed_df = df.copy()
    cols = [
        "is_sdk_or_data",
        "excerpt_data_or_sdk_extraction_json",
        "is_att",
        "excerpt_att_extraction_json"
    ]

    # cols = [
    #     "is_data_or_ai",
    #     "representative_quotes",
    #     "quantitative_summary",
    #     "qualitative_summary",
    #     "is_sdk_or_data",
    #     "excerpt_data_or_sdk_extraction_json",
    #     "is_att",
    #     "excerpt_att_extraction_json"
    # ]

    processed_df[cols] = None

    return processed_df


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    processed_df = load_dataframe(INPUT_FILE)
    results = asyncio.run(run_all(processed_df))

