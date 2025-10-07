import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, AuthenticationError, APIError
from io import StringIO, BytesIO
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # Fallback to heuristic if unavailable

# --- Constants ---
BATCH_SIZE = 64
MAX_CONCURRENCY = 8

# --- Pydantic Models ---
class Code(BaseModel):
    code: str = Field(..., description="The concise name of the code or theme.")
    description: str = Field(..., description="A clear, one-sentence explanation of what this code represents.")
    examples: list[str] = Field(default=[], description="A list of 3-5 verbatim example responses from the provided data that best illustrate this code.")

class Codebook(BaseModel):
    codes: list[Code] = Field(..., description="The complete list of generated codes for the survey question.")

# --- Classification output models (single and multi-label share the same shape) ---
class ClassificationEvidence(BaseModel):
    label: str = Field(..., description="The code label from the codebook")
    fragment: str = Field(..., description="The exact fragment of the response that supports this code")
    pertinence: float = Field(..., ge=0.0, le=1.0, description="Pertinence score between 0 and 1")
    explanation: str | None = Field(default=None, description="Brief explanation of why the fragment maps to the code")

class ClassificationOutput(BaseModel):
    items: list[ClassificationEvidence] = Field(default_factory=list, description="List of assigned codes with supporting evidence")

class UncoveredReview(BaseModel):
    uncovered: list[int] = Field(default_factory=list, description="0-based indices of responses with no applicable codes")

class BatchItem(BaseModel):
    index: int
    items: list[ClassificationEvidence] = Field(default_factory=list)

class BatchClassificationOutput(BaseModel):
    results: list[BatchItem] = Field(default_factory=list)

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent Survey Coder", page_icon="✨​", layout="wide")

# --- State Management ---
def initialize_state():
    for key, value in {
        'api_key': None, 'df': None, 'structured_codebook': None,
        'classified_df': None, 'question_text': "", 'initial_sample_size': 0,
        'codebook_upload_nonce': 0
    }.items():
        if key not in st.session_state: st.session_state[key] = value

initialize_state()

# --- Helper & API Functions ---
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): return pd.read_csv(uploaded_file, encoding='latin1')
        elif uploaded_file.name.endswith(('.xls', '.xlsx')): return pd.read_excel(uploaded_file)
    except Exception as e: st.error(f"Error loading file: {e}"); return None

@st.cache_data
def convert_df_to_downloadable(df, format="CSV"):
    if format == "CSV": return df.to_csv(index=False).encode('utf-8')
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()

# --- Token Estimation Helpers ---
def _get_token_encoder(model: str):
    if tiktoken is None:
        return None
    try:
        # Try exact model mapping first
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            # Heuristics for newer GPT-4.x / 4o families
            if any(k in model for k in ["gpt-4o", "gpt-4.1"]):
                return tiktoken.get_encoding("o200k_base")
            # Default for GPT-3.5/4 families
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

def estimate_token_count(text: str, model: str) -> int:
    encoder = _get_token_encoder(model)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Rough fallback: ~4 chars per token
    return max(1, int(len(text) / 4))

def estimate_chat_tokens(system_text: str, user_text: str, model: str) -> int:
    # Minimal overhead per message; we keep it simple and robust
    return estimate_token_count(system_text, model) + estimate_token_count(user_text, model) + 6

def reconstruct_codebook_text(codebook_obj: Codebook):
    if not codebook_obj or not codebook_obj.codes: return ""
    return "\n".join([f"- Code: {item.code}\n  Description: {item.description}" for item in codebook_obj.codes]).strip()

def generate_structured_codebook_prompt(question, examples):
    example_str = "\n".join([f'"{ex}"' for ex in examples])
    return f"""Analyze the survey question and responses to create a thematic codebook.
    **Question:** "{question}" **Responses:**\n[{example_str}]\n
    Identify themes, define a code and description for each, and select 3-5 verbatim examples. Include an "Other" code."""

def create_merge_prompt(codebook1_json: str, codebook2_json: str, user_instructions: str = "") -> str:
    prompt = f"""You are a master survey analyst consolidating two codebooks. Your goal is to create the most accurate final codebook.
    **Codebook A:**\n{codebook1_json}\n**Codebook B:**\n{codebook2_json}\n
    **Analytical Process:**
    1.  Identify codes with similar themes.
    2.  For similar codes, examine their `examples` and evaluate if it possible to separate the example into two more distinct code. If they are truly redundant, consolidate them.
    3.  Retain unique codes. Each code have to refer to an unique concept."""
    if user_instructions:
        prompt += f"""\n\n**CRITICAL USER INSTRUCTIONS:**\nYou MUST follow these instructions. They override general guidance.\n---\n{user_instructions}\n---"""
    return prompt

def classify_response_prompt(question, response, codebook_text, include_explanation: bool = True):
    explanation_field = ', "explanation": string' if include_explanation else ''
    explanation_note = '' if include_explanation else '\n    Do NOT include an "explanation" field.'
    return f"""Classify the response based on the codebook. Choose the single best code and provide evidence.
    Question: "{question}"
    Codebook:\n---\n{codebook_text}\n---
    Response: "{response}"
    Return ONLY a JSON object with this schema:
    {{
      "items": [
        {{ "label": string, "fragment": string, "pertinence": number (0-1){explanation_field} }}
      ]
    }}{explanation_note}
    For single-label, the list MUST contain exactly one item.
    """

# --- NEW: Prompt for multi-label classification ---
def classify_response_prompt_multi(question, response, codebook_text, include_explanation: bool = True):
    explanation_field = ', "explanation": string' if include_explanation else ''
    explanation_note = '' if include_explanation else '\n    Do NOT include an "explanation" field.'
    return f"""Analyze the response and identify ALL themes from the codebook that are present.
    Question: "{question}"
    Codebook:\n---\n{codebook_text}\n---
    Response: "{response}"
    Return ONLY a JSON object with this schema:
    {{
      "items": [
        {{ "label": string, "fragment": string, "pertinence": number (0-1){explanation_field} }}
      ]
    }}{explanation_note}
    If no codes apply, return {{ "items": [] }}.
    """

def get_embeddings(texts: list[str], api_key: str, model="text-embedding-3-small"):
    client = OpenAI(api_key=api_key)
    
    # Batch texts to ensure no batch exceeds 8k tokens
    MAX_TOKENS_PER_BATCH = 8000
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = estimate_token_count(text, model)
        
        # If adding this text would exceed limit, start a new batch
        if current_tokens + text_tokens > MAX_TOKENS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens
    
    # Add the last batch if it has content
    if current_batch:
        batches.append(current_batch)
    
    # Process each batch and collect all embeddings
    all_embeddings = []
    for batch in batches:
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([embedding.embedding for embedding in response.data])
    
    return all_embeddings

def call_openai_api(api_key, system_prompt, user_prompt, model="gpt-4o", pydantic_model=None):
    try:
        client = OpenAI(api_key=api_key)
        if pydantic_model:
            completion = client.chat.completions.parse(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], response_format=pydantic_model)
            return completion.choices[0].message.parsed
        else:
            response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
            return response.choices[0].message.content.strip()
    except Exception as e: st.error(f"API Error: {e}"); return None

# --- Codebook Import/Export Helpers ---
def codebook_to_json_bytes(codebook_obj: Codebook):
    try:
        return codebook_obj.model_dump_json(indent=2).encode('utf-8')
    except Exception as e:
        st.error(f"Failed to serialize codebook to JSON: {e}")
        return None

def codebook_to_csv_bytes(codebook_obj: Codebook):
    try:
        rows = []
        for item in codebook_obj.codes:
            rows.append({
                "code": item.code,
                "description": item.description,
                "examples": json.dumps(item.examples, ensure_ascii=False)
            })
        df = pd.DataFrame(rows, columns=["code", "description", "examples"])
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.error(f"Failed to serialize codebook to CSV: {e}")
        return None

def parse_uploaded_codebook(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.json'):
            data = json.load(uploaded_file)
            return Codebook.model_validate(data)
        elif name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
            if df is None or df.empty:
                return None
            normalized_map = {str(col).strip().lower(): col for col in df.columns}
            code_col = normalized_map.get('code') or normalized_map.get('label') or list(df.columns)[0]
            desc_col = normalized_map.get('description') or (list(df.columns)[1] if len(df.columns) > 1 else None)
            examples_col = normalized_map.get('examples')
            codes = []
            for _, row in df.iterrows():
                code_val = row.get(code_col)
                if pd.isna(code_val):
                    continue
                code_text = str(code_val).strip()
                if not code_text:
                    continue
                desc_text = ""
                if desc_col and desc_col in df.columns:
                    desc_val = row.get(desc_col)
                    if not pd.isna(desc_val):
                        desc_text = str(desc_val).strip()
                examples_list = []
                if examples_col and examples_col in df.columns:
                    cell = row.get(examples_col)
                    if not pd.isna(cell):
                        if isinstance(cell, str):
                            try:
                                parsed = json.loads(cell)
                                if isinstance(parsed, list):
                                    examples_list = [str(x) for x in parsed]
                                else:
                                    examples_list = [str(parsed)]
                            except json.JSONDecodeError:
                                for sep in ['|', ';', '\n']:
                                    if sep in cell:
                                        examples_list = [s.strip() for s in cell.split(sep) if s.strip()]
                                        break
                                if not examples_list and cell.strip():
                                    examples_list = [cell.strip()]
                        elif isinstance(cell, (list, tuple)):
                            examples_list = [str(x) for x in cell]
                codes.append(Code(code=code_text, description=desc_text, examples=examples_list))
            return Codebook(codes=codes)
    except Exception as e:
        st.error(f"Failed to parse codebook: {e}")
    return None

# --- Helpers for robust merging ---
def normalize_codebook(cb: Codebook) -> Codebook:
    try:
        normalized_codes = []
        for item in (cb.codes or []):
            code_label = str(getattr(item, 'code', '') or '').strip()
            code_desc = str(getattr(item, 'description', '') or '').strip()
            raw_examples = getattr(item, 'examples', []) or []
            if not isinstance(raw_examples, (list, tuple)):
                raw_examples = [str(raw_examples)]
            examples = [str(x).strip() for x in raw_examples if str(x).strip()]
            normalized_codes.append(Code(code=code_label, description=code_desc, examples=examples))
        return Codebook(codes=normalized_codes)
    except Exception:
        return cb

def serialize_codebook_for_prompt(codebook_obj: Codebook) -> str:
    try:
        payload = {
            "codes": [
                {
                    "code": c.code,
                    "description": c.description,
                    "examples": c.examples or []
                } for c in (codebook_obj.codes or [])
            ]
        }
        return json.dumps(payload, indent=2)
    except Exception:
        # Fallback to pydantic dump
        return codebook_obj.model_dump_json(indent=2)

def _extract_json_block(text: str) -> str:
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
    except Exception:
        pass
    return text

def merge_codebooks_via_llm(api_key: str, base_cb: Codebook, new_cb: Codebook, model: str, user_instructions: str):
    system_msg = "You are a master survey analyst."
    prompt = create_merge_prompt(
        serialize_codebook_for_prompt(base_cb),
        serialize_codebook_for_prompt(new_cb),
        user_instructions
    ) + "\n\nReturn ONLY a JSON object with this exact schema: { \"codes\": [ { \"code\": string, \"description\": string, \"examples\": string[] } ] }"
    # First try structured parsing
    merged = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=Codebook)
    if merged:
        return merged
    # Fallback to raw string and manual JSON parsing
    raw = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=None)
    if not raw:
        return None
    try:
        json_str = _extract_json_block(raw)
        data = json.loads(json_str)
        return Codebook.model_validate(data)
    except Exception as e:
        st.error(f"Failed to parse merged codebook: {e}")
        return None

def refine_codebook_via_instructions(api_key: str, current_cb: Codebook, instructions: str, model: str):
    system_msg = "You are a master survey analyst."
    base_json = serialize_codebook_for_prompt(current_cb)
    prompt = f"""You are refining an existing survey codebook strictly following the user's instructions.
Current codebook JSON:\n{base_json}\n\nInstructions:\n{instructions}\n\nReturn ONLY a JSON object with this exact schema: {{ \"codes\": [ {{ \"code\": string, \"description\": string, \"examples\": string[] }} ] }}. Do not add unrelated fields."""
    refined = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=Codebook)
    if refined:
        return refined
    raw = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=None)
    if not raw:
        return None
    try:
        json_str = _extract_json_block(raw)
        data = json.loads(json_str)
        return Codebook.model_validate(data)
    except Exception as e:
        st.error(f"Failed to parse refined codebook: {e}")
        return None

def propose_new_code_for_response(api_key: str, question: str, response: str, model: str) -> Code | None:
    system_msg = "You are an expert survey analyst. Propose concise, non-overlapping codes."
    prompt = f"""The following response does not match any existing code. Propose ONE new code.
Question: "{question}"
Response: "{response}"
Return ONLY JSON with this schema:
{{ "code": string, "description": string, "examples": string[] }}
Include the response as the first example.
"""
    proposed = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=Code)
    return proposed

def review_uncovered_responses(api_key: str, question: str, responses: list[str], codebook_text: str, model: str) -> list[int]:
    indexed = "\n".join([f"[{i}] \"{resp}\"" for i, resp in enumerate(responses)])
    system_msg = "You are a precise survey coding reviewer."
    prompt = f"""Identify which responses do NOT match any code in the codebook.
Question: "{question}"
Codebook:\n---\n{codebook_text}\n---
Responses (indexed):\n{indexed}\n\nReturn ONLY JSON with this schema and 0-based indices:
{{ "uncovered": [number, ...] }}
If all responses are covered, return {{ "uncovered": [] }}.
"""
    res = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=UncoveredReview)
    if res and isinstance(res.uncovered, list):
        return [i for i in res.uncovered if isinstance(i, int) and 0 <= i < len(responses)]
    return []

def _chunk_list(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i+size] for i in range(0, len(items), size)]

def classify_batch(api_key: str, question: str, responses: list[str], codebook_text: str, model: str, multi: bool, include_explanations: bool) -> list[dict]:
    if not responses:
        return []
    indexed = "\n".join([f"[{i}] \"{resp}\"" for i, resp in enumerate(responses)])
    system_msg = "You are a survey coding assistant."
    explanation_field = ', "explanation": string' if include_explanations else ''
    explanation_note = '' if include_explanations else '\nDo NOT include an "explanation" field.'
    single_rule = 'For single-label, each items list MUST contain exactly one item.' if not multi else ''
    prompt = f"""Analyze the indexed responses against the codebook.
Question: "{question}"
Codebook:\n---\n{codebook_text}\n---
Responses (indexed):\n{indexed}
Return ONLY JSON with this schema:
{{
  "results": [
    {{ "index": number, "items": [ {{ "label": string, "fragment": string, "pertinence": number (0-1){explanation_field} }} ] }}
  ]
}}{explanation_note}
{single_rule}
For uncovered responses, use an empty list for items.
"""
    parsed = call_openai_api(api_key, system_msg, prompt, model=model, pydantic_model=BatchClassificationOutput)
    # Build default empty aligned list
    aligned: list[dict] = [{"Assigned Code": "No Code Applied", "Details": []} for _ in responses]
    if not parsed or not parsed.results:
        return aligned
    for item in parsed.results:
        if not isinstance(item.index, int) or item.index < 0 or item.index >= len(responses):
            continue
        labels = [ev.label for ev in (item.items or [])]
        label_str = " | ".join(labels) if labels else "No Code Applied"
        details = [{
            "label": ev.label,
            "fragment": ev.fragment,
            "pertinence": ev.pertinence,
            "explanation": ev.explanation if include_explanations else None
        } for ev in (item.items or [])]
        aligned[item.index] = {"Assigned Code": label_str, "Details": details}
    return aligned

def classify_batches_async(api_key: str, question: str, batched_responses: list[list[str]], codebook_text: str, model: str, multi: bool, include_explanations: bool) -> list[list[dict]]:
    if not batched_responses:
        return []
    results: list[list[dict]] = [None] * len(batched_responses)  # type: ignore
    def worker(idx: int, batch: list[str]) -> tuple[int, list[dict]]:
        return idx, classify_batch(api_key, question, batch, codebook_text, model, multi, include_explanations)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        future_to_idx = {executor.submit(worker, i, b): i for i, b in enumerate(batched_responses)}
        for future in as_completed(future_to_idx):
            i, res = future.result()
            results[i] = res
    return results

# --- UI Layout ---
st.title("Survey Coder")
st.markdown("Generate, refine, merge, and efficiently classify survey data with AI.")

with st.sidebar:
    st.header("1. Setup")
    api_key_input = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key_input: st.session_state.api_key = api_key_input
    uploaded_file = st.file_uploader("Upload survey data", type=['csv', 'xlsx'])
    if uploaded_file and st.session_state.df is None:
        initialize_state(); st.session_state.api_key = api_key_input; st.session_state.df = load_data(uploaded_file)

if not st.session_state.api_key: st.warning("Please enter your OpenAI API key.")
elif st.session_state.df is None: st.info("Please upload a CSV or Excel file.")
else:
    # (Sections 2 and 3 are unchanged)
    df = st.session_state.df
    st.header("2. Configure Initial Coding Task")
    col_config_1, col_config_2 = st.columns(2)
    with col_config_1:
        valid_columns = []
        for col in df.columns:
            try:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    series = df[col].dropna().astype(str).str.strip()
                    unique_count = series[series != ""].nunique()
                    if unique_count > 50:
                        valid_columns.append(col)
            except Exception:
                continue
        if not valid_columns:
            st.warning("No text columns with > 50 unique non-empty values found.")
            st.stop()
        column_to_code = st.selectbox("Select column to code:", options=valid_columns)
        st.session_state.column_to_code = column_to_code
        st.session_state.question_text = st.text_area("Edit the question text:", value=column_to_code, height=100)
    with col_config_2:
        generation_model = st.selectbox("Select Model for Codebook Generation:", ["gpt-4.1", "gpt-4o",  "gpt-5"], help="A powerful model is recommended for generation and merging.")
        num_examples = st.slider("Examples for initial codebook:", 10, 600, 150, 10)

    st.divider()
    st.header("3. Generate & Refine Codebook")
    col_imp, col_gen = st.columns(2)
    with col_imp:
        with st.expander("📥 Import Codebook"):
            uploaded_cb = st.file_uploader(
                "Upload codebook (JSON or CSV)",
                type=['json', 'csv'],
                key=f"codebook_upload_{st.session_state.codebook_upload_nonce}"
            )
            import_clicked = st.button("Load Codebook", use_container_width=True)
            if import_clicked:
                if uploaded_cb is None:
                    st.warning("Please select a file to import.")
                else:
                    parsed_cb = parse_uploaded_codebook(uploaded_cb)
                    if parsed_cb and parsed_cb.codes:
                        parsed_cb = normalize_codebook(parsed_cb)
                        st.session_state.structured_codebook = parsed_cb
                        st.session_state.selected_code_index = 0
                        # Reset selector widget to avoid stale selection after import
                        try:
                            if 'code_selector' in st.session_state: del st.session_state['code_selector']
                        except Exception:
                            pass
                        st.success(f"Loaded codebook with {len(parsed_cb.codes)} codes.")
                        # Clear the uploader by changing its key
                        st.session_state.codebook_upload_nonce += 1
                        st.rerun()
                    else:
                        st.warning("Uploaded codebook is empty or invalid.")
    with col_gen:
        if st.button("✨ Generate Initial Codebook", use_container_width=True):
            st.session_state.initial_sample_size = num_examples
            examples = df[column_to_code].dropna().unique().tolist()[:num_examples]
            with st.spinner("AI is analyzing responses and generating your codebook..."):
                prompt = generate_structured_codebook_prompt(st.session_state.question_text, examples)
                codebook_object = call_openai_api(st.session_state.api_key, "You are an expert survey analyst.", prompt, generation_model, pydantic_model=Codebook)
                if codebook_object: st.session_state.structured_codebook = codebook_object; st.success("Initial codebook generated!")

    if st.session_state.structured_codebook:
        # Ensure imported or previously saved codebooks are normalized for editing
        st.session_state.structured_codebook = normalize_codebook(st.session_state.structured_codebook)
        # (Editor UI is unchanged)
        st.subheader("Interactive Codebook Editor")
        with st.expander("⬇️ Export Codebook"):
            json_bytes = codebook_to_json_bytes(st.session_state.structured_codebook)
            csv_bytes = codebook_to_csv_bytes(st.session_state.structured_codebook)
            dl1, dl2 = st.columns(2)
            with dl1:
                if json_bytes:
                    st.download_button("Download JSON", data=json_bytes, file_name="codebook.json", mime="application/json", use_container_width=True)
            with dl2:
                if csv_bytes:
                    st.download_button("Download CSV", data=csv_bytes, file_name="codebook.csv", mime="text/csv", use_container_width=True)
        code_labels = [item.code for item in st.session_state.structured_codebook.codes]
        if 'selected_code_index' not in st.session_state or st.session_state.selected_code_index >= len(code_labels): st.session_state.selected_code_index = 0
        selected_code_label = st.selectbox("Select a code to review and edit:", options=code_labels, key="code_selector", index=st.session_state.selected_code_index)
        selected_index = code_labels.index(selected_code_label) if selected_code_label in code_labels else -1

        if selected_index != -1:
            current_item = st.session_state.structured_codebook.codes[selected_index]
            col1, col2 = st.columns(2);
            with col1:
                st.markdown("#### Edit Code Details")
                current_item.code = st.text_input("Code Label", value=current_item.code, key=f"label_{selected_index}")
                current_item.description = st.text_area("Description", value=current_item.description, key=f"desc_{selected_index}", height=150)
                st.markdown("");
                if st.button("🗑️ Delete This Code", use_container_width=True):
                    remaining = [c for i, c in enumerate(st.session_state.structured_codebook.codes) if i != selected_index]
                    st.session_state.structured_codebook = Codebook(codes=remaining)
                    st.session_state.selected_code_index = 0
                    st.rerun()
            with col2:
                st.markdown(f"#### Examples for '{current_item.code}'")
                examples_text = "\n".join(current_item.examples) if current_item.examples else ""
                edited_examples_text = st.text_area(
                    "Edit examples (one per line)",
                    value=examples_text,
                    key=f"examples_editor_{selected_index}",
                    height=275
                )
                if st.button("💾 Save Codebook", key=f"save_codebook_{selected_index}", use_container_width=True):
                    # Update examples for the selected code from the textarea
                    new_list = [line.strip() for line in edited_examples_text.splitlines() if line.strip()]
                    current_item.examples = new_list
                    # Labels and descriptions are already bound via inputs in the left column
                    st.success("Codebook saved.")
                    st.rerun()

        with st.expander("➕ Add a New Code"):
            with st.form("new_code_form", clear_on_submit=True):
                new_code_label = st.text_input("New Code Label")
                new_code_desc = st.text_area("New Code Description")
                if st.form_submit_button("Add Code to Codebook"):
                    if new_code_label: st.session_state.structured_codebook.codes.append(Code(code=new_code_label, description=new_code_desc)); st.success(f"Added new code: '{new_code_label}'"); st.rerun()
                    else: st.warning("Please provide a label for the new code.")
        with st.expander("🔍 Review a New Sample for Uncovered Texts"):
            st.markdown("Sample responses not covered by existing labels and propose new codes where needed.")
            review_n = st.slider("Number of responses to review:", 5, 100, 20, 5)
            review_model = st.selectbox("Model for review:", ["gpt-4.1", "gpt-4o", "gpt-4.1-mini"], index=0, key="review_model")
            if st.button("Scan Sample for Uncovered Texts", use_container_width=True):
                col_to_use = st.session_state.get('column_to_code', None)
                if not col_to_use or col_to_use not in df.columns:
                    st.error("No valid column selected for coding.")
                else:
                    series = df[col_to_use].dropna().astype(str)
                    series = series[series.str.strip() != ""]
                    if series.empty:
                        st.warning("No responses available.")
                    else:
                        actual_n = min(review_n, len(series))
                        sample_list = series.sample(n=actual_n, replace=False).tolist()
                        cb_text = reconstruct_codebook_text(st.session_state.structured_codebook)
                        # Single prompt review of uncovered responses
                        uncovered_idx = review_uncovered_responses(
                            api_key=st.session_state.api_key,
                            question=st.session_state.question_text,
                            responses=sample_list,
                            codebook_text=cb_text,
                            model=review_model
                        )
                        st.info(f"Found {len(uncovered_idx)} uncovered responses out of {actual_n}.")
                        if uncovered_idx:
                            uncovered = [sample_list[i] for i in uncovered_idx]
                            st.markdown("Select a response to propose a new code:")
                            sel_resp = st.selectbox("Uncovered response:", options=uncovered, key="uncovered_select")
                            if st.button("Propose New Code", use_container_width=True):
                                with st.spinner("Proposing a new code..."):
                                    proposed = propose_new_code_for_response(st.session_state.api_key, st.session_state.question_text, sel_resp, review_model)
                                    if proposed and proposed.code:
                                        st.session_state.structured_codebook.codes.append(proposed)
                                        st.success(f"Added proposed code: {proposed.code}")
                                        st.rerun()
                                    else:
                                        st.error("Failed to propose a new code.")
        
        with st.expander("📝 Refine with Instructions (no new examples)"):
            user_refine_instructions = st.text_area("Write instructions to refine the current codebook:", placeholder="e.g., 'Combine \"Delivery Time\" and \"Shipping Speed\". Split \"Price\" into \"High Price\" and \"Unexpected Fees\".'", height=140)
            if st.button("✨ Apply Instructional Refinement"):
                if not user_refine_instructions.strip():
                    st.warning("Please provide instructions to refine the codebook.")
                else:
                    with st.spinner("Applying your instructions to refine the codebook..."):
                        refined = refine_codebook_via_instructions(
                            api_key=st.session_state.api_key,
                            current_cb=st.session_state.structured_codebook,
                            instructions=user_refine_instructions,
                            model=generation_model
                        )
                        if not refined:
                            st.error("Failed to refine the codebook with the provided instructions.")
                        else:
                            st.session_state.structured_codebook = normalize_codebook(refined)
                            st.session_state.selected_code_index = 0
                            st.success("Codebook refined using your instructions.")
                            st.rerun()

        with st.expander("🔄 Refine codebook automatically with new examples"):
            st.markdown("Generate a second codebook from a new random sample and merge it with the current one.")
            refine_sample_size = st.slider("Number of examples to resample:", 10, 600, 150, 10)
            user_merge_instructions = st.text_area("Additional Instructions for Merging (Optional):", placeholder="e.g., 'Be more specific with the codes'", height=120)
            if st.button("🚀 Refine & Merge Codebook"):
                with st.spinner("Refining and merging codebook..."):
                    initial_codebook = st.session_state.structured_codebook
                    all_unique_responses = df[column_to_code].dropna().unique()
                    if len(all_unique_responses) == 0:
                        st.warning("No responses available to sample for refinement.")
                    else:
                        actual_sample_size = min(len(all_unique_responses), refine_sample_size)
                        new_examples = pd.Series(all_unique_responses).sample(n=actual_sample_size, replace=False).tolist()
                        new_prompt = generate_structured_codebook_prompt(st.session_state.question_text, new_examples)
                        new_codebook = call_openai_api(st.session_state.api_key, "You are an expert survey analyst.", new_prompt, generation_model, pydantic_model=Codebook)
                        if not new_codebook:
                            st.error("Failed to generate the refinement codebook.")
                        else:
                            merged_codebook = merge_codebooks_via_llm(
                                api_key=st.session_state.api_key,
                                base_cb=initial_codebook,
                                new_cb=new_codebook,
                                model=generation_model,
                                user_instructions=user_merge_instructions
                            )
                            if not merged_codebook:
                                st.error("Failed to merge codebooks.")
                            else:
                                st.session_state.structured_codebook = normalize_codebook(merged_codebook)
                                st.session_state.selected_code_index = 0
                                st.success("Codebooks merged! The updated codebook is now displayed below.")
                                st.rerun()

        st.divider()
        st.header("4. Test Codebook")
        st.markdown("##### Classify Custom Text")
        manual_text = st.text_area("Enter a response to classify:", height=120)
        test_model = st.selectbox("Model for testing:", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0, key="test_model_single")
        test_multilabel = st.checkbox("Enable Multi-Label for test", value=False, key="test_multilabel_single")
        if st.button("Classify Text", use_container_width=True):
            final_codebook_text = reconstruct_codebook_text(st.session_state.structured_codebook)
            if not manual_text.strip():
                st.warning("Please enter some text to classify.")
            elif not final_codebook_text:
                st.error("Codebook is empty.")
            else:
                if test_multilabel:
                    prompt = classify_response_prompt_multi(st.session_state.question_text, manual_text.strip(), final_codebook_text)
                    parsed = call_openai_api(st.session_state.api_key, "You are a multi-label survey coding assistant.", prompt, model=test_model, pydantic_model=ClassificationOutput)
                else:
                    prompt = classify_response_prompt(st.session_state.question_text, manual_text.strip(), final_codebook_text)
                    parsed = call_openai_api(st.session_state.api_key, "You are a survey coding assistant.", prompt, model=test_model, pydantic_model=ClassificationOutput)
                if not parsed or not parsed.items:
                    st.info("No Code Applied")
                else:
                    items_df = pd.DataFrame([{
                        "label": it.label,
                        "fragment": it.fragment,
                        "pertinence": it.pertinence,
                        "explanation": it.explanation
                    } for it in parsed.items])
                    st.dataframe(items_df, use_container_width=True)

        st.markdown("##### Classify Random Sample")
        sample_n = st.slider("Number of random responses:", 1, 50, 10, 1)
        test_model_batch = st.selectbox("Model for testing batch:", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0, key="test_model_batch")
        test_multilabel_batch = st.checkbox("Enable Multi-Label for batch", value=False, key="test_multilabel_batch")
        if st.button("Classify Random Sample", use_container_width=True):
            col_to_use = st.session_state.get('column_to_code', None)
            if not col_to_use or col_to_use not in df.columns:
                st.error("No valid column selected for coding.")
            else:
                series = df[col_to_use].dropna().astype(str)
                series = series[series.str.strip() != ""]
                if series.empty:
                    st.warning("No responses available to classify.")
                else:
                    actual_n = min(sample_n, len(series))
                    sample_list = series.sample(n=actual_n, replace=False).tolist()
                    final_codebook_text = reconstruct_codebook_text(st.session_state.structured_codebook)
                    if not final_codebook_text:
                        st.error("Codebook is empty.")
                    else:
                        evidence_rows = []
                        for resp in sample_list:
                            if test_multilabel_batch:
                                prompt = classify_response_prompt_multi(st.session_state.question_text, resp, final_codebook_text)
                                parsed = call_openai_api(st.session_state.api_key, "You are a multi-label survey coding assistant.", prompt, model=test_model_batch, pydantic_model=ClassificationOutput)
                            else:
                                prompt = classify_response_prompt(st.session_state.question_text, resp, final_codebook_text)
                                parsed = call_openai_api(st.session_state.api_key, "You are a survey coding assistant.", prompt, model=test_model_batch, pydantic_model=ClassificationOutput)
                            if not parsed or not parsed.items:
                                evidence_rows.append({
                                    "Response": resp,
                                    "label": "No Code Applied",
                                    "fragment": "",
                                    "pertinence": None,
                                    "explanation": ""
                                })
                            else:
                                for it in parsed.items:
                                    evidence_rows.append({
                                        "Response": resp,
                                        "label": it.label,
                                        "fragment": it.fragment,
                                        "pertinence": it.pertinence,
                                        "explanation": it.explanation
                                    })
                        test_df = pd.DataFrame(evidence_rows)
                        st.dataframe(test_df, use_container_width=True)
        
        st.divider()
        st.header("5. Classify All Responses")
        classification_model = st.selectbox("Select Model for Final Classification:", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0)
        
        # --- NEW: Checkboxes for classification mode ---
        col_mode_1, col_mode_2, col_mode_3 = st.columns(3)
        with col_mode_1:
            use_multilabel = st.checkbox("✅ Enable Multi-Label Classification", value=False, help="Allow assigning multiple codes to a single response. More comprehensive but can be slower.")
        with col_mode_2:
            use_clustering = st.checkbox("⚡️ Accelerate with Semantic Clustering", value=True, help="Group similar responses to reduce API calls. Highly recommended.")
        with col_mode_3:
            include_explanations = st.checkbox("💬 Include Explanations", value=True, help="If disabled, the model will not generate explanations to save tokens.")

        # Pre-calculate an estimated token usage for a single classification call
        try:
            preview_example = "example response"
            final_codebook_text_preview = reconstruct_codebook_text(st.session_state.structured_codebook)
            if use_multilabel:
                prompt_preview = classify_response_prompt_multi(st.session_state.question_text, preview_example, final_codebook_text_preview, include_explanation=include_explanations)
                system_preview = "You are a multi-label survey coding assistant."
            else:
                prompt_preview = classify_response_prompt(st.session_state.question_text, preview_example, final_codebook_text_preview, include_explanation=include_explanations)
                system_preview = "You are a survey coding assistant."
            est_tokens_per_call = estimate_chat_tokens(system_preview, prompt_preview, classification_model)
        except Exception:
            est_tokens_per_call = None

        st.caption(f"Estimated tokens per classification call: {est_tokens_per_call if est_tokens_per_call is not None else 'N/A'}")

        # Estimated embedding tokens (pre-clustering)
        try:
            base_unique_responses = df[column_to_code].dropna().unique().tolist()
            string_responses = [str(item) for item in base_unique_responses]
            unique_responses = [text for text in string_responses if text.strip()]
            if use_clustering:
                embedding_model = "text-embedding-3-small"
                total_embed_tokens = sum(estimate_token_count(resp, embedding_model) for resp in unique_responses)
            else:
                total_embed_tokens = 0
            st.caption(f"Estimated embedding tokens (pre-clustering): {total_embed_tokens}")
        except Exception:
            pass

        # Estimated total classification tokens across mini-batches (upper bound if clustering is enabled)
        try:
            final_codebook_text_preview = reconstruct_codebook_text(st.session_state.structured_codebook)
            # Build batched prompts as classify_batch does
            batches_for_est = _chunk_list(unique_responses, BATCH_SIZE)
            total_class_tokens = 0
            for batch in batches_for_est:
                indexed = "\n".join([f"[{i}] \"{resp}\"" for i, resp in enumerate(batch)])
                explanation_field = ', "explanation": string' if include_explanations else ''
                explanation_note = '' if include_explanations else '\nDo NOT include an "explanation" field.'
                single_rule = 'For single-label, each items list MUST contain exactly one item.' if not use_multilabel else ''
                user_text = f"""Analyze the indexed responses against the codebook.
Question: "{st.session_state.question_text}"
Codebook:\n---\n{final_codebook_text_preview}\n---
Responses (indexed):\n{indexed}
Return ONLY JSON with this schema:
{{
  "results": [
    {{ "index": number, "items": [ {{ "label": string, "fragment": string, "pertinence": number (0-1){explanation_field} }} ] }}
  ]
}}{explanation_note}
{single_rule}
For uncovered responses, use an empty list for items.
"""
                system_text = "You are a survey coding assistant."
                total_class_tokens += estimate_chat_tokens(system_text, user_text, classification_model)
            if use_clustering:
                st.caption(f"Estimated total classification tokens (no-clustering upper bound): {total_class_tokens}")
            else:
                st.caption(f"Estimated total classification tokens: {total_class_tokens}")
        except Exception:
            pass

        if st.button("🚀 Classify All Responses", use_container_width=True):
            final_codebook_text = reconstruct_codebook_text(st.session_state.structured_codebook)
            if not final_codebook_text: st.error("Codebook is empty.")
            else:
                #unique_responses = df[column_to_code].dropna().unique().tolist()

                # 1. Get potentially mixed-type unique values
                base_unique_responses = df[column_to_code].dropna().unique().tolist()
                                
                # 2. Convert every item to a string first
                string_responses = [str(item) for item in base_unique_responses]
                                
                # 3. Now, safely filter out empty/whitespace strings
                unique_responses = [text for text in string_responses if text.strip()]

                results_cache = {}
                progress_bar = st.progress(0, text="Initializing classification...")
                # Show an overall rough estimate of total prompt tokens before starting
                try:
                    if use_clustering and len(unique_responses) > 1:
                        # Upper bound: cluster reps + outlier batch count; conservative estimate
                        # We cannot know ahead of time; display per-call estimate only
                        st.info(f"Per-call token estimate: ~{est_tokens_per_call if est_tokens_per_call is not None else 'N/A'} tokens")
                    else:
                        batches = _chunk_list(unique_responses, BATCH_SIZE)
                        total_calls = len(batches)
                        total_est = est_tokens_per_call * total_calls if est_tokens_per_call is not None else None
                        st.info(f"Estimated total prompt tokens: {total_est if total_est is not None else 'N/A'} (across {total_calls} calls)")
                except Exception:
                    pass
                
                # --- MODIFIED: The core classification loop now handles multi-label ---
                def classify_item(response):
                    if use_multilabel:
                        prompt = classify_response_prompt_multi(st.session_state.question_text, response, final_codebook_text, include_explanation=include_explanations)
                        parsed = call_openai_api(st.session_state.api_key, "You are a multi-label survey coding assistant.", prompt, model=classification_model, pydantic_model=ClassificationOutput)
                    else: # Single-label path
                        prompt = classify_response_prompt(st.session_state.question_text, response, final_codebook_text, include_explanation=include_explanations)
                        parsed = call_openai_api(st.session_state.api_key, "You are a survey coding assistant.", prompt, model=classification_model, pydantic_model=ClassificationOutput)
                    if not parsed or parsed.items is None:
                        return {"Assigned Code": "API_ERROR", "Details": []}
                    labels = [item.label for item in parsed.items] if parsed.items else []
                    label_str = " | ".join(labels) if labels else "No Code Applied"
                    details = [{
                        "label": item.label,
                        "fragment": item.fragment,
                        "pertinence": item.pertinence,
                        "explanation": item.explanation if include_explanations else None
                    } for item in parsed.items]
                    return {"Assigned Code": label_str, "Details": details}

                if use_clustering and len(unique_responses) > 1:
                    # (Clustering logic remains the same, but now calls the unified classify_item function)
                    progress_bar.progress(5, text="Step 1/4: Generating embeddings..."); embeddings = get_embeddings(unique_responses, st.session_state.api_key)
                    if not embeddings: st.error("Failed to generate embeddings."); st.stop()
                    progress_bar.progress(15, text="Step 2/4: Clustering responses..."); embeddings = normalize(np.array(embeddings)); db = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings); labels = db.labels_
                    cluster_ids = set(labels); n_clusters = len(cluster_ids) - (1 if -1 in labels else 0); outliers = [response for response, label in zip(unique_responses, labels) if label == -1]; n_outliers = len(outliers)
                    outlier_batches = _chunk_list(outliers, BATCH_SIZE)
                    total_api_calls = n_clusters + len(outlier_batches)
                    if total_api_calls == 0: st.info("No new responses to classify."); st.stop()
                    st.info(f"Found {n_clusters} groups and {n_outliers} unique outliers. Total classifications needed: {total_api_calls}.")
                    calls_made = 0; response_to_cluster = {response: label for response, label in zip(unique_responses, labels)}; classified_clusters = {}
                    outlier_status = st.empty()
                    for cluster_id in cluster_ids:
                        if cluster_id != -1:
                            representative = next(response for response, label in response_to_cluster.items() if label == cluster_id)
                            code_str = classify_item(representative) # Call unified function
                            classified_clusters[cluster_id] = code_str; calls_made += 1
                            progress_bar.progress(15 + int(70 * (calls_made / total_api_calls)), text=f"Step 3/4: Classifying group {calls_made}/{total_api_calls}...")
                    # Minibatch outliers (async)
                    if outliers:
                        batched = outlier_batches
                        outlier_status.info(f"Launching {len(batched)} outlier batches asynchronously (up to {MAX_CONCURRENCY} concurrent)...")
                        async_results = classify_batches_async(
                            api_key=st.session_state.api_key,
                            question=st.session_state.question_text,
                            batched_responses=batched,
                            codebook_text=final_codebook_text,
                            model=classification_model,
                            multi=use_multilabel,
                            include_explanations=include_explanations
                        )
                        for batch_index, (batch, batch_results) in enumerate(zip(batched, async_results), start=1):
                            for resp, res in zip(batch, batch_results):
                                results_cache[resp] = res
                            calls_made += 1
                            progress_bar.progress(15 + int(70 * (calls_made / total_api_calls)), text=f"Step 3/4: Completed outlier batch {batch_index}/{len(batched)}")
                    outlier_status.empty()
                    for response, label in response_to_cluster.items():
                        if label != -1: results_cache[response] = classified_clusters[label]
                else:
                    # Minibatch the full set
                    batches = _chunk_list(unique_responses, BATCH_SIZE)
                    total_batches = len(batches)
                    processed = 0
                    batch_status = st.empty()
                    batch_status.info(f"Launching {total_batches} batches asynchronously (up to {MAX_CONCURRENCY} concurrent)...")
                    async_results = classify_batches_async(
                        api_key=st.session_state.api_key,
                        question=st.session_state.question_text,
                        batched_responses=batches,
                        codebook_text=final_codebook_text,
                        model=classification_model,
                        multi=use_multilabel,
                        include_explanations=include_explanations
                    )
                    for batch_index, (batch, batch_results) in enumerate(zip(batches, async_results), start=1):
                        for resp, res in zip(batch, batch_results):
                            results_cache[resp] = res
                        processed += len(batch)
                        progress_bar.progress(int(100 * processed / len(unique_responses)), text=f"Classifying unique responses {processed}/{len(unique_responses)} (batch {batch_index}/{total_batches})...")
                    batch_status.empty()
                
                progress_bar.progress(95, text="Step 4/4: Applying classifications...")
                final_df = df.copy()
                # Map label strings and details separately
                assigned_map = {k: v.get("Assigned Code", "") for k, v in results_cache.items()}
                details_map = {k: v.get("Details", []) for k, v in results_cache.items()}
                final_df['Assigned Code'] = final_df[column_to_code].map(assigned_map)
                final_df['Assigned Details'] = final_df[column_to_code].map(details_map)
                st.session_state.classified_df = final_df
                progress_bar.progress(100, text="Classification complete!"); st.success("Classification complete!")

    if st.session_state.classified_df is not None:
        st.divider()
        st.header("5. View and Download Results")
        # Normalize separator to pipe for multi-label results (backward compatibility with older runs)
        # Note: This conversion is disabled to preserve commas within single code labels
        # If you have old data with comma-separated labels, manually convert them to pipe-separated format
        # if 'Assigned Code' in st.session_state.classified_df.columns:
        #     try:
        #         series = st.session_state.classified_df['Assigned Code']
        #         if pd.api.types.is_object_dtype(series):
        #             needs_conversion = series.fillna("").str.contains(",").any()
        #             if needs_conversion:
        #                 st.session_state.classified_df['Assigned Code'] = series.fillna("").str.replace(r'\s*,\s*', ' | ', regex=True)
        #     except Exception:
        #         pass
        # Only show the original coded column and the assigned code
        col_to_show = st.session_state.get('column_to_code', None)
        if not col_to_show:
            try:
                col_to_show = column_to_code
            except Exception:
                col_to_show = None
        cols = []
        if col_to_show and col_to_show in st.session_state.classified_df.columns:
            cols.append(col_to_show)
        if 'Assigned Code' in st.session_state.classified_df.columns:
            cols.append('Assigned Code')
        display_df = st.session_state.classified_df[cols] if cols else st.session_state.classified_df
        # Optional filter to view only one label in the table
        try:
            assigned_series = st.session_state.classified_df['Assigned Code'].fillna("")
            all_labels_view = sorted({
                c.strip()
                for row in assigned_series
                for c in re.split(r'\s*\|\s*', str(row))
                if c.strip() and c not in ("No Code Applied", "API_ERROR")
            })
        except Exception:
            all_labels_view = []
        selected_label_filter = st.selectbox(
            "Filter table by label",
            options=["All labels"] + all_labels_view,
            index=0,
            key="results_label_filter"
        )
        if selected_label_filter != "All labels" and 'Assigned Code' in display_df.columns:
            mask = st.session_state.classified_df['Assigned Code'].fillna("").apply(
                lambda s: selected_label_filter in [c.strip() for c in re.split(r'\s*\|\s*', str(s)) if c.strip()]
            )
            display_df = display_df[mask]
        st.dataframe(display_df)

        # Show details for selected row
        if 'Assigned Details' in st.session_state.classified_df.columns:
            with st.expander("View classification details"):
                idx = st.number_input("Row index to inspect", min_value=0, max_value=len(st.session_state.classified_df)-1, value=0, step=1)
                details = st.session_state.classified_df.iloc[int(idx)]['Assigned Details']
                if isinstance(details, list) and details:
                    details_df = pd.DataFrame(details)
                    st.dataframe(details_df, use_container_width=True)
                else:
                    st.write("No details available for this row.")
        
        # --- MODIFIED: Frequency table now handles multi-label results ---
        st.subheader("Code Frequencies")
        freq_df = st.session_state.classified_df['Assigned Code'].dropna()
        # Check if we need to split multi-label strings
        if use_multilabel:
            # Split pipe-separated strings into lists, then create a new row for each code
            freq_df = freq_df.str.split(r'\s*\|\s*').explode()
        
        freq_counts = freq_df.value_counts().reset_index()
        freq_counts.columns = ['Code', 'Frequency']
        freq_counts['Percentage'] = (freq_counts['Frequency'] / freq_counts['Frequency'].sum()).map('{:.2%}'.format)
        st.dataframe(freq_counts, use_container_width=True)

        # Prepare one-hot (multi-label) CSV
        ohe_bytes = None
        try:
            if 'Assigned Code' in st.session_state.classified_df.columns:
                ser = st.session_state.classified_df['Assigned Code'].fillna("")
                # Parse labels per row using pipe separator
                lists = ser.apply(lambda s: [c.strip() for c in re.split(r'\s*\|\s*', s) if c.strip()] if isinstance(s, str) and s else [])
                # Build unique set of labels excluding placeholders
                all_labels = sorted({c for sub in lists for c in sub if c not in ("No Code Applied", "API_ERROR")})
                if all_labels:
                    ohe_df = pd.DataFrame(0, index=st.session_state.classified_df.index, columns=all_labels, dtype=int)
                    for idx, labels in lists.items():
                        for c in labels:
                            if c in ohe_df.columns:
                                ohe_df.at[idx, c] = 1
                    # Prepend original column if available
                    if col_to_show and col_to_show in st.session_state.classified_df.columns:
                        ohe_df = pd.concat([st.session_state.classified_df[[col_to_show]], ohe_df], axis=1)
                    ohe_bytes = convert_df_to_downloadable(ohe_df, "CSV")
        except Exception as e:
            pass

        # Prepare simplified CSV with just original text and code
        simple_csv_bytes = None
        try:
            if 'Assigned Code' in st.session_state.classified_df.columns:
                col_to_show = st.session_state.get('column_to_code', None)
                if not col_to_show:
                    try:
                        col_to_show = column_to_code
                    except Exception:
                        col_to_show = None
                
                if col_to_show and col_to_show in st.session_state.classified_df.columns:
                    simple_df = st.session_state.classified_df[[col_to_show, 'Assigned Code']].copy()
                    simple_df.columns = ['Original Text', 'Assigned Code']
                    simple_csv_bytes = convert_df_to_downloadable(simple_df, "CSV")
        except Exception as e:
            pass

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.download_button("📥 Download as CSV", convert_df_to_downloadable(st.session_state.classified_df, "CSV"), "classified_data.csv", "text/csv", use_container_width=True)
        d_col2.download_button("📥 Download as Excel", convert_df_to_downloadable(st.session_state.classified_df, "Excel"), "classified_data.xlsx", use_container_width=True)
        d_col3.download_button("📥 Download One-Hot (CSV)", ohe_bytes if ohe_bytes else b"", "classified_one_hot.csv", "text/csv", disabled=(ohe_bytes is None), use_container_width=True)
        d_col4.download_button("📥 Download Simple (CSV)", simple_csv_bytes if simple_csv_bytes else b"", "simple_classified.csv", "text/csv", disabled=(simple_csv_bytes is None), use_container_width=True)

        with st.expander("♻️ Reclassify by Included Labels"):
            if not st.session_state.structured_codebook or not st.session_state.structured_codebook.codes:
                st.warning("No codebook available.")
            else:
                all_labels = [c.code for c in st.session_state.structured_codebook.codes]
                included = st.multiselect(
                    "Include only these labels for reclassification (rows without any of these labels are kept as-is):",
                    options=all_labels,
                    key="reclass_included_rows"
                )
                re_model = st.selectbox("Model:", ["gpt-4.1-mini", "gpt-4.1-nano"], index=0, key="reclass_rows_model")
                re_multilabel = st.checkbox("Enable Multi-Label", value=True, key="reclass_rows_multilabel")
                re_explanations = st.checkbox("Include Explanations", value=True, key="reclass_rows_explanations")
                include_no_code_rows = st.checkbox("Include rows currently labeled 'No Code Applied'", value=True, key="reclass_include_no_code")
                if st.button("Reclassify Selected Rows", use_container_width=True):
                    if col_to_show is None or col_to_show not in st.session_state.classified_df.columns:
                        st.error("No valid text column available.")
                    else:
                        df_view = st.session_state.classified_df
                        indices_to_reclassify = []
                        texts_to_reclassify = []
                        included_set = set(included)
                        for idx, row in df_view.iterrows():
                            assigned = str(row.get('Assigned Code', '') or '')
                            labels = [c.strip() for c in re.split(r'\s*\|\s*', assigned) if c.strip() and c not in ("No Code Applied", "API_ERROR")]
                            if len(labels) == 0:
                                if not include_no_code_rows:
                                    continue
                                indices_to_reclassify.append(idx)
                                texts_to_reclassify.append(str(row[col_to_show]))
                            else:
                                # Include semantics: reclassify if the row has ANY label in the included set
                                if len(included_set) == 0:
                                    # Nothing selected: by default, do not reclassify labeled rows
                                    continue
                                if re_multilabel:
                                    if any(l in included_set for l in labels):
                                        indices_to_reclassify.append(idx)
                                        texts_to_reclassify.append(str(row[col_to_show]))
                                else:
                                    # Single-label: the one label must be included
                                    if labels[0] in included_set:
                                        indices_to_reclassify.append(idx)
                                        texts_to_reclassify.append(str(row[col_to_show]))
                        if not indices_to_reclassify:
                            st.info("No rows require reclassification based on the selected exclusions.")
                        else:
                            cb_text = reconstruct_codebook_text(st.session_state.structured_codebook)
                            batches = _chunk_list(texts_to_reclassify, BATCH_SIZE)
                            total_batches = len(batches)
                            progress_bar = st.progress(0, text="Reclassifying in batches...")
                            batch_status = st.empty()
                            batch_status.info(f"Launching {total_batches} batches asynchronously (up to {MAX_CONCURRENCY} concurrent)...")
                            async_results = classify_batches_async(
                                api_key=st.session_state.api_key,
                                question=st.session_state.question_text,
                                batched_responses=batches,
                                codebook_text=cb_text,
                                model=re_model,
                                multi=re_multilabel,
                                include_explanations=re_explanations
                            )
                            # Flatten results and apply to the selected indices
                            flat_results = [res for batch_res in async_results for res in batch_res]
                            for i, (row_idx, res) in enumerate(zip(indices_to_reclassify, flat_results), start=1):
                                df_view.at[row_idx, 'Assigned Code'] = res.get('Assigned Code', '')
                                df_view.at[row_idx, 'Assigned Details'] = res.get('Details', [])
                                if total_batches > 0:
                                    progress_bar.progress(int(100 * i / len(indices_to_reclassify)), text=f"Reclassifying rows {i}/{len(indices_to_reclassify)} (batch progress)")
                            st.session_state.classified_df = df_view
                            progress_bar.progress(100, text="Reclassification complete!")
                            st.success("Reclassification complete! Rows with only excluded labels were kept as-is; others were reclassified with the current codebook.")
                            try:
                                cols_to_show = [col_to_show, 'Assigned Code'] if col_to_show in df_view.columns else ['Assigned Code']
                                st.markdown("#### Reclassified Rows")
                                st.dataframe(df_view.loc[indices_to_reclassify, cols_to_show], use_container_width=True)
                            except Exception:
                                pass