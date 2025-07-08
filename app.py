import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import plotly.graph_objects as go
from datetime import datetime

# Initialize Flask
app = Flask(__name__)
CORS(app)

# === ENVIRONMENT VARIABLES ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DB_HOST = os.getenv("DB_HOST", "db.diiskfbsryethkjsoobv.supabase.internal")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "credable@123")  # Ideally set via secrets
DB_NAME = os.getenv("DB_NAME", "postgres")
PINECONE_INDEX_NAME = "applicationjson"

# === OpenAI & Pinecone Setup ===
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

# === DB Utility ===
def get_db_connection():
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME
        )
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/healthcheck")
def health():
    return "OK", 200

@app.route("/save_correction", methods=["POST"])
def save_correction():
    data = request.get_json()
    correction_text = data.get("correction", "").strip()
    query_id = data.get("query_id")

    if not correction_text or not query_id:
        return jsonify({"error": "Correction text and Query ID required"}), 400

    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "DB connection failed"}), 500

    try:
        cursor = db_conn.cursor()
        cursor.execute("UPDATE log_1 SET correction = %s WHERE id = %s;", (correction_text, query_id))
        db_conn.commit()
        cursor.close()
        return jsonify({"message": "Correction saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db_conn.close()

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    query = data.get("query")
    application_id = data.get("application_id")

    if not query or not application_id:
        return jsonify({"error": "Missing query or application_id"}), 400

    # Vectorize
    try:
        embed = client.embeddings.create(model="text-embedding-ada-002", input=query)
        vector = embed.data[0].embedding
    except Exception as e:
        return jsonify({"error": f"OpenAI embedding failed: {str(e)}"}), 500

    # Search Pinecone
    try:
        pinecone_results = pinecone_index.query(vector=vector, top_k=50, include_metadata=True).get("matches", [])
    except Exception as e:
        return jsonify({"error": f"Pinecone query failed: {str(e)}"}), 500

    context = construct_context(pinecone_results)
    history_context = build_history(application_id)

    # Generate LLM response
    answer = generate_answer(query, context, history_context)

    # Store in DB
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "DB connection failed"}), 500
    try:
        cursor = db_conn.cursor()
        meta = json.dumps({"matched_files": serialize_pinecone_results(pinecone_results)})
        cursor.execute("""
            INSERT INTO log_1 (query, application_id, response, metadata)
            VALUES (%s, %s, %s, %s);
        """, (query, application_id, answer, meta))
        cursor.execute("SELECT LASTVAL();")
        query_id = cursor.fetchone()[0]
        db_conn.commit()
        cursor.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db_conn.close()

    return jsonify({"query": query, "response": answer, "query_id": query_id})


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


# === Helper Functions ===

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

def build_history(application_id):
    db_conn = get_db_connection()
    if not db_conn:
        return ""

    try:
        cursor = db_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT query, response FROM log_1
            WHERE application_id = %s
            ORDER BY id DESC
            LIMIT 5
        """, (application_id,))
        logs = cursor.fetchall()
        return "\n\n".join([f"Q: {log['query']}\nA: {log['response']}" for log in logs[::-1]])
    except Exception as e:
        return ""
    finally:
        db_conn.close()

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

# === Entry Point for Render ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
