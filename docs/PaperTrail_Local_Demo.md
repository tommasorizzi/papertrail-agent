# Run PaperTrail locally

From the project root, install the locked dependencies and start Streamlit.

```sh
uv sync --locked
uv run streamlit run app.py
```

PaperTrail starts in **Scripted demo** with an answered example. The other two
questions show a partial answer with a follow-up search and an insufficient
evidence result. Citation links jump to evidence cards for the bundled
synthetic passages.

## Enable live mode

Set `OPENAI_API_KEY` in the environment before starting Streamlit. To load it
from the project's ignored `.env` file, launch with
`uv run --env-file .env streamlit run app.py`.

Select **Live**, enter a question, and submit the form. PaperTrail uses
`gpt-4.1-mini` by default. Set `OPENAI_MODEL` to use another OpenAI model
that supports Structured Outputs. The adapter sets a 20-second SDK timeout
and disables SDK retries.

Live mode sends the submitted question and selected passages to OpenAI. Use
non-sensitive questions with the bundled synthetic collection.
