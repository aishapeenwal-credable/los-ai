import os
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, jsonify, send_file
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import json
from datetime import datetime
from flask_cors import CORS
import plotly.io as pio
import plotly.graph_objects as go

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Hardcoded API Keys
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_6EVcM3_7NfQ5yHqWnK66LpJmMszMGvVmeWMpSPNySd1sw2EkJ59S9e7n6iLdcYH8eBTGoa"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "applicationjson"

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Database Connection
def get_db_connection():
    try:
        db_conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "db.diiskfbsryethkjsoobv.supabase.co"),
            port=os.getenv("port", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "credable@123"),
            dbname=os.getenv("DB_NAME", "postgres")
        )
        return db_conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None


# Serialize Pinecone Results
def serialize_pinecone_results(results):
    return [
        {"id": result.id, "score": result.score, "metadata": result.metadata}
        for result in results
    ]
# Construct Context
def construct_context(pinecone_results):
    if not pinecone_results:
        return "No relevant data available in context."

    context_lines = []
    for result in pinecone_results:
        metadata = result.get("metadata", {})
        for key, value in metadata.items():
            context_lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(context_lines)


# Get Last 5 Logs for Chain of Thought
def get_last_five_logs(application_id):
    db_conn = get_db_connection()
    if db_conn is None:
        return []

    try:
        cursor = db_conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT query, response 
            FROM log_1 
            WHERE application_id = %s 
            ORDER BY id DESC 
            LIMIT 1;
        """
        cursor.execute(query, (application_id,))
        logs = cursor.fetchall()
        cursor.close()
        return logs[::-1]
    except psycopg2.Error as e:
        print(f"Database error fetching logs: {e}")
        return []
    finally:
        db_conn.close()

# Vectorize Query
def get_query_vector(query):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating query vector: {e}")
        return None

# Patch for ChatCompletion (use OpenAI v1+ format)
def chat_completion(messages, model="gpt-4.5-preview", max_tokens=2000):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return "An error occurred while generating the response."

# Generate Answer
def answer_query_with_openai(query, context, history_context=""):
    prompt = f"""
You are a senior credit underwriter assessing a business loan application in the Indian regulatory and financial context. Use the chain of thought technique by analyzing current and past queries to deepen your understanding. Analyze the application using only explicitly provided data—no assumptions or hypotheticals.

---

Conversation History:
{history_context if history_context else "No prior queries."}

---

Relevant Context:
{context}

---

New Query:
{query}

Answer:
"""
    messages = [
        {"role": "system", "content": "You are a financial assistant."},
        {"role": "user", "content": prompt}
    ]
    return chat_completion(messages)

# Save Correction
def update_correction_in_db(query_id, correction):
    db_conn = get_db_connection()
    if db_conn is None:
        return {"error": "Database connection unavailable"}

    try:
        cursor = db_conn.cursor()
        update_query = """
            UPDATE log_1 
            SET correction = %s
            WHERE id = %s;
        """
        cursor.execute(update_query, (correction, query_id))
        db_conn.commit()
        cursor.close()
        return {"message": "Correction updated successfully"}
    except psycopg2.Error as e:
        return {"error": str(e)}
    finally:
        db_conn.close()

@app.route("/save_correction", methods=["POST"])
def save_correction():
    data = request.get_json()
    correction_text = data.get("correction", "").strip()
    query_id = data.get("query_id")

    if not correction_text:
        return jsonify({"error": "Correction text is required"}), 400
    if not query_id:
        return jsonify({"error": "Query ID is required"}), 400

    try:
        result = update_correction_in_db(query_id, correction_text)
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Query Handler
@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        query = data.get("query")
        application_id = data.get("application_id")

        if not query:
            return jsonify({"error": "Query is required"}), 400
        if not application_id:
            return jsonify({"error": "Application ID is required"}), 400

        query_vector = get_query_vector(query)
        if not query_vector:
            return jsonify({"error": "Failed to generate query vector"}), 500

        pinecone_results = pinecone_index.query(
            vector=query_vector,
            top_k=50,
            include_metadata=True
        ).get("matches", [])

        if not pinecone_results:
            return jsonify({"error": "No relevant data found"}), 404

        context = construct_context(pinecone_results)
        conversation_history = get_last_five_logs(application_id)
        history_lines = [f"Q{idx+1}: {log_1['query']}\nA{idx+1}: {log_1['response']}" for idx, log_1 in enumerate(conversation_history)]
        history_context = "\n\n".join(history_lines)

        response = answer_query_with_openai(query, context, history_context)

        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        metadata_json = json.dumps({"matched_files": serialize_pinecone_results(pinecone_results)}) if pinecone_results else None
        query_str = """
            INSERT INTO log_1 (query, application_id, response, metadata)
            VALUES (%s, %s, %s, %s);
        """
        cursor.execute(query_str, (query, application_id, response, metadata_json))
        db_conn.commit()
        query_id = cursor.lastrowid
        cursor.close()
        db_conn.close()

        return jsonify({"query": query, "response": response, "query_id": query_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

# Visualization Endpoint
@app.route('/visualise', methods=['POST'])
def visualise_response():
    data = request.get_json()
    response_text = data.get("response", "")

    if not response_text:
        return jsonify({"error": "Response text required"}), 400

    visualization_prompt = f"""
You are a data visualization expert. Analyze the following response text carefully:

{response_text}

Decide if this data can be meaningfully visualized using a single Plotly graph. 

If yes:
- Clearly state "YES"
- Provide explicit Plotly Python JSON instructions in the 'plotly_instruction' field including title, axis labels, and legend.

If visualization does not apply or isn't clear:
- Clearly state "NO"
- Return an empty object for 'plotly_instruction'.

Respond in the following strict JSON format:
{{
  "can_visualize": "YES or NO",
  "plotly_instruction": {{...}}
}}
"""

    try:
        messages = [
            {"role": "system", "content": "You're a visualization assistant."},
            {"role": "user", "content": visualization_prompt}
        ]
        llm_response = chat_completion(messages, max_tokens=1500).strip()

        # Normalize Markdown-wrapped JSON (e.g., ```json\n{...}\n```)
        if llm_response.startswith("```json"):
            llm_response = llm_response.replace("```json", "").replace("```", "").strip()
        elif llm_response.startswith("```"):
            llm_response = llm_response.replace("```", "").strip()

        # Optional debug logging
        print("Cleaned LLM Response:", repr(llm_response))

        try:
            parsed_response = json.loads(llm_response)
        except json.JSONDecodeError as e:
            return jsonify({
                "error": "Failed to parse JSON from LLM response",
                "raw_response": llm_response,
                "debug": str(e)
            }), 500


        # DEBUG: Log raw response
        print("Raw LLM Response:", repr(llm_response))

        # Ensure the response is valid JSON
        try:
            parsed_response = json.loads(llm_response)
        except json.JSONDecodeError as e:
            return jsonify({
                "error": "Malformed LLM response",
                "raw_response": llm_response,
                "debug": str(e)
            }), 500

        can_visualize = parsed_response.get("can_visualize", "NO") == "YES"
        plotly_instruction = parsed_response.get("plotly_instruction", {}) if can_visualize else {}

        return jsonify({
            "can_visualize": can_visualize,
            "plotly_instruction": plotly_instruction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Run Flask App
if __name__ == "__main__":
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
