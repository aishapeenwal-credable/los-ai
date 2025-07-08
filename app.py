import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

# === Flask App ===
app = Flask(__name__)
CORS(app)

# === ENV VARS ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PINECONE_INDEX_NAME = "applicationjson"

# === Clients ===
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/healthcheck")
def health():
    return "OK", 200

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    application_id = data.get("application_id")

    if not query or not application_id:
        return jsonify({"error": "Missing query or application_id"}), 400

    # Vectorize query
    try:
        embed = client.embeddings.create(model="text-embedding-ada-002", input=query)
        vector = embed.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"OpenAI embedding failed: {str(e)}"}), 500

    # Query Pinecone
    try:
        pinecone_results = pinecone_index.query(vector=vector, top_k=50, include_metadata=True).get("matches", [])
    except Exception as e:
        return jsonify({"error": f"Pinecone query failed: {str(e)}"}), 500

    context = construct_context(pinecone_results)
    history_context = fetch_supabase_logs(application_id)
    response = generate_answer(query, context, history_context)

    metadata_json = json.dumps({"matched_files": serialize_pinecone_results(pinecone_results)}) if pinecone_results else None
    inserted = insert_supabase_log(query, application_id, response, metadata_json)
    query_id = inserted[0]['id'] if inserted else None

    return jsonify({"query": query, "response": response, "query_id": query_id})

@app.route("/save_correction", methods=["POST"])
def save_correction():
    data = request.get_json()
    correction = data.get("correction")
    query_id = data.get("query_id")

    if not correction or not query_id:
        return jsonify({"error": "Missing correction or query_id"}), 400

    updated = update_supabase_correction(query_id, correction)
    return jsonify(updated)

@app.route("/visualise", methods=["POST"])
def visualise_response():
    data = request.get_json()
    response_text = data.get("response", "")

    if not response_text:
        return jsonify({"error": "Missing response"}), 400

    prompt = f"""
You are a data visualization expert. Analyze the following response:

{response_text}

Can you visualize it? If yes, respond in JSON with:
{{"can_visualize": "YES", "plotly_instruction": {{...}}}}
Otherwise, respond:
{{"can_visualize": "NO", "plotly_instruction": {{}}}}
"""

    messages = [
        {"role": "system", "content": "You are a visualization assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        raw_response = chat_completion(messages, max_tokens=1500)
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw_response)
    except Exception as e:
        return jsonify({"error": f"Failed to parse response: {e}", "raw_response": raw_response}), 500

    return jsonify({
        "can_visualize": parsed.get("can_visualize") == "YES",
        "plotly_instruction": parsed.get("plotly_instruction", {})
    })

# === Supabase REST API ===
def insert_supabase_log(query, application_id, response, metadata_json):
    url = f"{SUPABASE_URL}/rest/v1/log_1"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    payload = {
        "query": query,
        "application_id": application_id,
        "response": response,
        "metadata": metadata_json
    }
    r = requests.post(url, headers=headers, json=payload)
    return r.json()

def fetch_supabase_logs(application_id):
    url = f"{SUPABASE_URL}/rest/v1/log_1?application_id=eq.{application_id}&order=id.desc&limit=5"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return ""
    logs = r.json()
    logs.reverse()
    return "\n\n".join([f"Q: {log['query']}\nA: {log['response']}" for log in logs])

def update_supabase_correction(query_id, correction):
    url = f"{SUPABASE_URL}/rest/v1/log_1?id=eq.{query_id}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.patch(url, headers=headers, json={"correction": correction})
    if r.status_code == 204:
        return {"message": "Correction updated successfully."}
    return {"error": r.text}

# === Utility Functions ===
def serialize_pinecone_results(results):
    return [{"id": r.id, "score": r.score, "metadata": r.metadata} for r in results]

def construct_context(results):
    if not results:
        return "No relevant context found."
    context_lines = []
    for r in results:
        meta = r.get("metadata", {})
        for k, v in meta.items():
            context_lines.append(f"{k.replace('_', ' ').title()}: {v}")
    return "\n".join(context_lines)

def generate_answer(query, context, history):
    prompt = f"""
You are a senior credit underwriter assessing business loan applications in India. Use the provided context and past conversation history (if any). Avoid assumptions.

History:
{history or "None"}

Context:
{context}

Question:
{query}

Answer:
"""
    messages = [
        {"role": "system", "content": "You are a financial assistant."},
        {"role": "user", "content": prompt}
    ]
    return chat_completion(messages)

def chat_completion(messages, model="gpt-4.5-preview", max_tokens=2000):
    try:
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"

# === Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
