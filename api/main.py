import uvicorn
import psycopg2
import ollama
import pypdf
import nltk
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import contextmanager
import io
import os  # Importante

# --- Configuración ---
# Lee las variables de entorno de Docker
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin")
DB_NAME = os.getenv("DB_NAME", "rag_poc")

DB_PARAMS = {
    "host": DB_HOST, "port": 5432, "user": DB_USER,
    "password": DB_PASS, "database": DB_NAME
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Modelos
EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "llama3:70b"  # Ajusta si usas otro (ej: llama3:8b)
VECTOR_DIMENSION = 1024  # Dimensión de bge-m3

# Cliente de Ollama que apunta al contenedor
client = ollama.Client(host=OLLAMA_HOST)

# NLTK (para chunking)
nltk.download('punkt', quiet=True)

app = FastAPI(title="AI Backend PoC")


# --- Base de Datos ---

@contextmanager
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        yield conn
    except Exception as e:
        print(f"Error de conexión a la base de datos: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor (BD)")
    finally:
        if 'conn' in locals():
            conn.close()


def setup_database():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk TEXT NOT NULL,
                embedding VECTOR({VECTOR_DIMENSION})
            );
            """)
            conn.commit()
    print("Base de datos configurada y lista.")


# --- Endpoints ---

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    try:
        pdf_bytes = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))

        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() or ""

        chunks = nltk.sent_tokenize(full_text)

        print(f"Procesando '{file.filename}'. {len(chunks)} chunks generados.")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    if not chunk.strip(): continue

                    response = client.embeddings(
                        model=EMBEDDING_MODEL,
                        prompt=chunk
                    )
                    embedding = response["embedding"]

                    cur.execute(
                        "INSERT INTO documents (filename, chunk, embedding) VALUES (%s, %s, %s)",
                        (file.filename, chunk, embedding)
                    )
                conn.commit()

        return {"status": "success", "filename": file.filename, "chunks_stored": len(chunks)}
    except Exception as e:
        print(f"Error procesando el PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {e}")


class QueryRequest(BaseModel):
    pregunta: str
    top_k: int = 5


@app.post("/generate")
async def generate_response(request: QueryRequest):
    try:
        response = client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=request.pregunta
        )
        query_embedding = response["embedding"]

        context_chunks = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk FROM documents ORDER BY embedding <-> %s::vector LIMIT %s",
                    (str(query_embedding), request.top_k)
                )
                results = cur.fetchall()
                context_chunks = [row[0] for row in results]

        if not context_chunks:
            return {"respuesta": "No encontré información relevante.", "contexto": []}

        context_str = "\n\n".join(context_chunks)
        prompt = f"""
        Usando SÓLO el siguiente contexto, responde la pregunta.
        Contexto:
        ---
        {context_str}
        ---
        Pregunta: {request.pregunta}
        """

        print(f"Enviando prompt a {GENERATION_MODEL}...")

        ollama_response = client.chat(
            model=GENERATION_MODEL,
            messages=[
                {'role': 'system', 'content': 'Responde basándote estrictamente en el contexto.'},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )

        respuesta = ollama_response['message']['content']

        return {"respuesta": respuesta, "contexto": context_chunks}
    except Exception as e:
        print(f"Error en la generación: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la generación: {e}")


# --- Inicio ---
if __name__ == "__main__":
    # La configuración de la BD la maneja el contenedor al iniciar
    print("Iniciando servidor FastAPI en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # Asegurarse de que la BD se configure cuando uvicorn importa el módulo
    setup_database()