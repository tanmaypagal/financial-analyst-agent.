"""Streamlit chat UI. Calls the same agent.core.run_agent as the REST API (no second implementation).

Run:  streamlit run ui/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent.config import load_personas, load_sectors, valid_sectors  # noqa: E402
from agent.core import AgentOutputError, run_agent_sync  # noqa: E402

st.set_page_config(page_title="Financial Analyst Agent", layout="wide")
personas, sectors = load_personas(), load_sectors()

with st.sidebar:
    st.header("Analyst setup")
    persona = st.selectbox("Persona", list(personas), format_func=lambda k: personas[k]["display_name"])
    sector = st.selectbox("Sector", valid_sectors(), format_func=lambda k: sectors[k].get("display_name", k))
    st.caption(personas[persona]["lens_description"].strip())
    if st.button("Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()

st.title(f"{personas[persona]['display_name']} - {sectors[sector].get('display_name', sector)}")
st.caption("Answers come only from the SQLite database via MCP tools. The last 3 exchanges are sent as context so follow-ups work; every number is re-fetched. "
           "Rows are marked verified only where a filing check passed (see the README).")
history = st.session_state.setdefault("history", [])


def details(r: dict):
    with st.expander(f"Details - confidence: {r['confidence']} - {len(r['tools_called'])} tool calls - {r['latency_ms']/1000:.1f}s"):
        st.markdown("**Confidence reasons**\n" + "\n".join(f"- {x}" for x in r["confidence_reasons"]))
        if r["data_gaps"]:
            st.markdown("**Data gaps**\n" + "\n".join(f"- {x}" for x in r["data_gaps"]))
        st.markdown("**Persona output**")
        st.json(r["persona_output"], expanded=False)
        st.markdown("**Data points**")
        st.dataframe(pd.DataFrame(r["data_points"]), use_container_width=True)
        st.markdown("**Tools called**")
        st.dataframe(pd.DataFrame(r["tools_called"]), use_container_width=True)
        st.caption(f"model: {r['model']} - companies: {', '.join(r['companies_referenced'])}")


for m in history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("resp"):
            details(m["resp"])

if q := st.chat_input("Ask about the sector..."):
    history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Querying data via MCP..."):
            try:
                prior = [{"query": history[i]["content"], "answer": history[i + 1]["content"]}      # last 3 completed Q/A turns
                         for i in range(0, len(history) - 1, 2) if history[i]["role"] == "user" and history[i + 1].get("resp")]   # skip failed turns
                resp = run_agent_sync(q, persona, sector, history=prior[-3:]).model_dump()
                history.append({"role": "assistant", "content": resp["answer"], "resp": resp})
                st.markdown(resp["answer"])
                details(resp)
            except (RuntimeError, AgentOutputError) as e:
                msg = f"Could not answer: {e}"
                history.append({"role": "assistant", "content": msg})
                st.error(msg)
