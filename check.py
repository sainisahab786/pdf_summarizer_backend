import os
import sqlite3
import openai
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY=os

# Initialize FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS Configuration
origins = [
    "http://localhost:3000", 
    "http://localhost:3001",  
    "https://yourdomain.com",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins + ["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
    expose_headers=["Access-Control-Allow-Origin"], 
)

# **Directly set OpenAI API Key**
OPENAI_API_KEY = "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"

# SQLite Database Path
DB_PATH = "users.db"

# Request Models
class QueryRequest(BaseModel):
    user_query: str
    age_filter: Optional[int] = None  
    location_filter: Optional[str] = None  

class RawQueryRequest(BaseModel):
    sql_query: str

class AnalyzeRequest(BaseModel):
    results: List[Dict]

# Function to Initialize Database
def init_db():
    if not os.path.exists(DB_PATH):  
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            location TEXT
        )
        """)

        sample_data = [
            ("Alice", 55, "New York"),
            ("Bob", 42, "Los Angeles"),
            ("Charlie", 63, "Chicago"),
            ("David", 58, "San Francisco"),
            ("Emma", 47, "Boston")
        ]

        cursor.executemany("INSERT INTO users (name, age, location) VALUES (?, ?, ?)", sample_data)
        conn.commit()
        conn.close()
        print("Database initialized with sample data.")

# Function to Generate SQL Query using OpenAI
def generate_sql_query(user_query, age_filter=None, location_filter=None):
    prompt = f"""
    Convert this natural language query into an SQL query for SQLite3.
    The table is named 'users' with columns: id (INTEGER), name (TEXT), age (INTEGER), location (TEXT).

    User query: '{user_query}'
    """

    if age_filter:
        prompt += f"\nFilter: Age greater than {age_filter}."
    if location_filter:
        prompt += f"\nFilter: Location = '{location_filter}'."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY  # Use OpenAI key directly
        )

        sql_query = response["choices"][0]["message"]["content"]
        return sql_query.strip()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None

# Function to Execute SQL Query
def execute_sql_query(sql_query):
    try:
        connection = sqlite3.connect(DB_PATH)
        cursor = connection.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row)) for row in results]
    except Exception as e:
        print(f"SQL Execution Error: {e}")
        return []
    finally:
        connection.close()

import os

def generate_graph(results):
    if not results:
        return None

    # Extract numerical data for visualization
    ages = [row["age"] for row in results if "age" in row]
    names = [row["name"] for row in results if "name" in row]

    if not ages:
        return None  # No numerical data to plot

    # Ensure the static directory exists
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(names, ages, color='skyblue')
    plt.xlabel("Users")
    plt.ylabel("Age")
    plt.title("Age Distribution of Users")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_path = os.path.join(static_dir, "analysis_graph.png")
    plt.savefig(img_path)
    plt.close()
    
    return img_path


# API Endpoint for Raw SQL Queries
@app.post("/api/raw_query")
async def execute_raw_query(request: RawQueryRequest):
    try:
        connection = sqlite3.connect(DB_PATH)
        cursor = connection.cursor()

        if not request.sql_query.strip().lower().startswith("select"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")

        cursor.execute(request.sql_query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]

        formatted_results = [dict(zip(column_names, row)) for row in results]

        return {"query": request.sql_query, "results": formatted_results}
    except sqlite3.Error as e:
        print(f"SQL Execution Error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL Execution Error: {str(e)}")
    finally:
        connection.close()

# API Endpoint to Fetch Available Filters
@app.get("/api/filters")
async def get_filters():
    filters = [
        {"name": "Age", "options": ["greater than", "less than", "equal to"]},
        {"name": "Location", "options": []},
    ]
    return {"filters": filters}

# API Endpoint for Chatbot Query Processing
@app.post("/api/query")
async def chatbot(request: QueryRequest):
    sql_query = generate_sql_query(request.user_query, request.age_filter, request.location_filter)
    
    if not sql_query:
        return {"error": "Failed to generate SQL query."}

    results = execute_sql_query(sql_query)

    return {"query": sql_query, "results": results}

# API Endpoint to Analyze Data
@app.post("/api/analyze")
async def analyze_data(request: AnalyzeRequest):
    results = request.results

    if not results:
        return {"analysis": "No data available for analysis."}

    data_summary = "\n".join([str(row) for row in results[:5]])

    prompt = f"""
    Analyze the following dataset and provide key insights:
    {data_summary}

    Identify patterns, trends, or anomalies in the data. Provide useful observations such as:
    - Common trends
    - Unusual or interesting insights
    - Any statistical significance
    - Future recommendations

    Please summarize in 3-5 bullet points.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            api_key=OPENAI_API_KEY  # Use OpenAI key directly
        )

        analysis_text = response["choices"][0]["message"]["content"]
        return {"analysis": analysis_text}
    
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analysis.")

# API Endpoint to Analyze Data and Generate Graph
@app.post("/api/analyze_with_graph")
async def analyze_data_with_graph(request: AnalyzeRequest):
    results = request.results

    if not results:
        return {"analysis": "No data available for analysis.", "graph_url": None}

    analysis = await analyze_data(request)
    
    graph_url = generate_graph(results)

    return {"analysis": analysis["analysis"], "graph_url": f"http://localhost:8000/{graph_url}" if graph_url else None}

# Initialize Database on Startup
init_db()
