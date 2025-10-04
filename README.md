# Notes Q&A Agent (Streamlit)

Ask natural-language questions over your local `.txt` / `.md` notes using semantic search (Sentence Transformers).

## 🚀 Quick start (local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the URL it prints (usually http://localhost:8501).

## 📂 Add your notes
Place your `.txt` or `.md` files in the `docs/` folder. The app auto-indexes them and shows how many files/chunks it has loaded. Use the **Rebuild index** button after adding/removing files.

## ☁️ Deploy to Streamlit Community Cloud
1. Create a new GitHub repo, e.g. `notes-qa-streamlit`.
2. Add these files:
   - `app.py`
   - `requirements.txt`
   - `docs/` (with your notes) – optional initially
3. Go to https://share.streamlit.io/ and connect your GitHub account.
4. Choose the repo and set **Main file** to `app.py`.
5. Deploy.

## ⚙️ How it works
- Splits each document into ~900-character chunks.
- Encodes chunks with `all-MiniLM-L6-v2` (fast, accurate).
- Finds nearest chunks via cosine similarity (`NearestNeighbors`).
- Generates an extractive answer from the top chunks and shows sources.

## 🔒 Privacy
All processing happens in the Streamlit app. No external APIs are used.

## 🧪 Sample data
This repo includes a `docs/` folder with a couple of small sample FAQs so you can test immediately.
