# ReturnRisk

ReturnRisk helps shoppers check whether they can return an item. It looks up the retailer’s official policy, asks for missing details such as the delivery date or item condition, and links to the policy behind its assessment. You can also try a different return date and switch back to your original purchase.

## Tools

| Tool | Version used | Role |
| --- | --- | --- |
| Python | 3.12 | Conversation logic, policy checks, and deadline calculations |
| Streamlit | 1.67.0 | Chat interface |
| Google Gen AI SDK | 2.34.0 | Interprets messages and extracts policy rules using Gemini |
| Brave Search API | Web Search API v1 | Finds candidate policy pages |
| Requests | 2.34.2 | Calls Brave and downloads retailer pages |
| Beautiful Soup | 4.15.0 | Extracts readable text from HTML |
| Pydantic | 2.13.4 | Validates purchase details and model output |
| tldextract | 5.4.0 | Checks whether pages belong to the retailer’s domain |
| python-dotenv | 1.2.2 | Loads local API keys |
| urllib3 | 2.8.0 | Supports the HTTP connection adapter through Requests |
| tzdata | 2026.5 | Provides timezone data on Windows |

`requirements.txt` lists the supported dependency ranges. Development checks use pytest 9.2.0 and Streamlit’s AppTest.

## Run locally

Install Python 3.12 and have an internet connection. Get a [Gemini API key](https://aistudio.google.com/apikey) and a [Brave Search API key](https://api-dashboard.search.brave.com/).

Open the project folder in VS Code and run these commands in the PowerShell terminal.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Open `.env` in VS Code and fill in `GEMINI_API_KEY` and `BRAVE_API_KEY`. The example file sets `GEMINI_MODEL` to `gemini-3.5-flash-lite`. Keep `.env` local and commit only `.env.example`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` in your browser. Press `Ctrl+C` in the terminal to stop the app.

## Try it

> I bought Olay face cream from an Ulta store on October 1, 2026. It is gently used, I have the receipt, and I have all the original components. Can I return it on October 20, 2026?

Then ask, “What if I return it on November 15, 2026?” Expand **Official policy quotation** to see how the policy distinguishes a refund from merchandise credit.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Runs the Streamlit interface and stores chat state. |
| `chatbot.py` | Interprets messages, updates purchase details, and formats replies. |
| `engine.py` | Searches and fetches policy pages, checks extracted rules against the source text, and evaluates eligibility. |
| `.env.example` | Shows the configuration fields to fill in locally. |
| `requirements.txt` | Lists the packages needed to run the app. |

The app uses official pages retrieved during research as its document source. It can reuse recently retrieved evidence within the session. **Refresh Policy** requests a fresh lookup. If a source is blocked or a rule cannot be confirmed, the app reports the uncertainty.

## License

ReturnRisk is available under the [MIT License](LICENSE).
