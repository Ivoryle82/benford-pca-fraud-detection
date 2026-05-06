import os
from typing import Dict

try:
    import cohere
except Exception:
    cohere = None
try:
    import streamlit as st
except Exception:
    st = None


def build_explanation_prompt(customer_data: Dict, population_context: Dict) -> str:
    """Builds the richer explanation prompt provided by the user.

    Expects `customer_data` and `population_context` formatted as in the
    user's example. Returns a single string prompt ready to send to the
    model.
    """
    # Defensive defaults
    cd = customer_data.copy()
    pc = population_context.copy()

    # Provide N/A display for missing items
    benford_time = cd.get("benford_time") if cd.get("benford_time") is not None else "N/A"
    top_repeats = cd.get("top_repeated_amounts") or []
    top_repeats_s = "; ".join([f"{a} (x{c})" for a, c in top_repeats]) if top_repeats else "None provided"

    prompt = f"""You are a fraud risk analyst at a payments company explaining a customer's anomaly score to a non-technical reviewer.

CRITICAL CONTEXT — read this before writing anything:
This system uses Benford's Law applied per-user across transaction amounts, inter-transaction time deltas, and amount-to-mean ratios. Empirical analysis on the IEEE-CIS dataset showed an inverted pattern relative to typical forensic-accounting Benford applications:

- HIGH Excess MAD (>0.030) typically indicates LEGITIMATE customer behavior dominated by subscription-style recurring charges at psychological price points ($9.99, $14.99, $59.00). This is NOT a fraud signal.

- LOW Excess MAD (close to 0) on a high-volume customer can indicate automated card-testing patterns — bots iterating through varied amounts that incidentally approximate a Benford distribution.

- MEDIUM Excess MAD (0.010-0.020) is the normal range for diverse legitimate spending without heavy subscription contamination.

So this score is a transaction-pattern fingerprint, not a fraud verdict. You explain what the *pattern* looks like, not whether it's fraud.

CUSTOMER DATA:
- Transaction count: {cd.get('n_txn', 'N/A')}
- Excess MAD on amounts: {cd.get('benford_amt', 0):.4f} ({cd.get('benford_amt_percentile', 0):.0f}th percentile)
- Excess MAD on ratios: {cd.get('benford_ratio', 0):.4f} ({cd.get('benford_ratio_percentile', 0):.0f}th percentile)
- Excess MAD on time deltas: {benford_time}
- Top repeated amounts: {top_repeats_s}

POPULATION BASELINE: {pc.get('n_users_in_population', 'N/A')} customers with typical legitimate Excess MAD on amounts in the range {pc.get('typical_legit_benford_amt', 'N/A')}.

TASK: Write a 2-3 sentence explanation of this customer's pattern. Cover:
1. What the score looks like relative to the population
2. The most likely behavioral interpretation (subscription-heavy, diverse spending, bot-like, etc.)
3. What a reviewer should look at next

CONSTRAINTS:
- Do NOT use the words "fraud" or "fraudulent" as a verdict — say "anomalous pattern" or "signal" instead
- Do NOT cite specific dollar amounts unless provided in the input
- Do NOT recommend account actions (block, freeze, etc.) — that's the reviewer's call
- Use plain English, not statistical jargon — assume the reader knows business but not statistics

Output format: just the 2-3 sentences. No preamble, no headers."""

    return prompt


def generate_explanation(customer_data: Dict, population_context: Dict, max_tokens: int = 120) -> str:
    """Generate a short plain-English explanation for a flagged transaction.

    Preference order for API key: `COHERE_API_KEY` env var, then Streamlit
    `st.secrets['COHERE_API_KEY']` (if Streamlit is available). Gracefully
    falls back to a deterministic message when the SDK/key is missing.
    Uses the Cohere Chat API and falls back to older generation paths if
    necessary.
    """
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key and st is not None:
        try:
            api_key = st.secrets.get("COHERE_API_KEY")
        except Exception:
            api_key = api_key

    prompt = build_explanation_prompt(customer_data, population_context)

    # Fallback when API key / SDK missing
    if not api_key or cohere is None:
        return (
            "This transaction is flagged because the leading digit distribution "
            "for this merchant/category deviates from the expected Benford pattern, "
            "suggesting potential manipulation or structured activity."
        )

    try:
        client = cohere.Client(api_key)

        # Prefer Chat API (new). Build simple system + user messages.
        messages = [
            {"role": "system", "content": "You are an assistant that explains why a transaction was flagged as anomalous. Provide a concise plain-English explanation (1-2 sentences)."},
            {"role": "user", "content": prompt},
        ]

        # Use chat.create if available
        if hasattr(client, "chat"):
            resp = None
            try:
                resp = client.chat.create(
                    model="command-xlarge-nightly",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
            except Exception:
                resp = None

            # Try a few response shapes for compatibility
            if resp is not None:
                text = None
                try:
                    text = resp.choices[0].message.content
                except Exception:
                    pass
                if not text:
                    try:
                        text = resp.output[0].content
                    except Exception:
                        pass
                if not text:
                    try:
                        text = resp.generations[0].text
                    except Exception:
                        pass
                if text:
                    return text.strip()

        # Fallback to legacy generate if chat not present or didn't return
        try:
            resp = client.generate(
                model="command-xlarge-nightly",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            text = resp.generations[0].text.strip()
            return text
        except Exception:
            pass

        return (
            "Could not generate explanation: unexpected response from Cohere API."
        )
    except Exception as e:
        # Surface helpful migration hint if present in the error message
        msg = str(e)
        if "Generate API was removed" in msg or "migrating-from-cogenerate-to-cochat" in msg:
            return (
                "Could not generate explanation: Cohere Generate API was removed. "
                "Please migrate to the Cohere Chat API and ensure your SDK is up-to-date. "
                "See https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat for details."
            )
        return f"Could not generate explanation: {e}"
