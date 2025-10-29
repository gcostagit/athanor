import uvicorn
import psycopg2
import ollama
import pypdf
import nltk
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import contextmanager
import io
import os

# --- Configuración (Lectura de Variables de Entorno) --------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin")
DB_NAME = os.getenv("DB_NAME", "rag_poc")

DB_PARAMS = {
    "host": DB_HOST, "port": 5432, "user": DB_USER,
    "password": DB_PASS, "database": DB_NAME
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# MODELOS ESTRATÉGICOS: BGE-M3 para embedding, Mistral para 32K tokens de memoria
EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "mistral:7b"  # <--- ¡CORRECCIÓN! Usamos Mistral 7B para los 32K tokens
VECTOR_DIMENSION = 1024

client = ollama.Client(host=OLLAMA_HOST)
nltk.download('punkt', quiet=True)

app = FastAPI(title="Athanor AI Backend")


# --- Lógica de Conexión y Base de Datos (Mantenida) ---

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


# --- ENDPOINT DE INGESTA (/upload) (Mantenido) ---

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    try:
        pdf_bytes = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))

        full_text = ""
        for page in pdf_reader.pages: full_text += page.extract_text() or ""

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


# --- ENDPOINT DE GENERACIÓN (/generate) (Corregido el Rol) ---

class QueryRequest(BaseModel):
    pregunta: str
    top_k: int = 5


@app.post("/generate")
async def generate_response(request: QueryRequest):
    try:
        # 1. Vectorizar la pregunta (con BGE-M3)
        response = client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=request.pregunta
        )
        query_embedding = response["embedding"]

        # 2. Buscar en pgvector los chunks más relevantes
        context_chunks = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk FROM documents ORDER BY embedding <-> %s::vector LIMIT %s",
                    (str(query_embedding), request.top_k)
                )
                context_chunks = [row[0] for row in cur.fetchall()]

        if not context_chunks:
            return {
                "respuesta": "Lo siento, la información para responder a esta pregunta no se encuentra en el material de estudio que tengo indexado.",
                "contexto": []}

        # 3. Definición del Rol (SYSTEM PROMPT 'Athanor Tutor')
        context_str = "\n\n".join(context_chunks)
        system_prompt = """
        Eres 'Athanor Tutor', un experto de IA diseñado para generar exámenes y planes de estudio.
        Tu rol es estricto, formal, objetivo y siempre debes basarte SÓLO en la información contenida en la sección CONTEXTO.
        Si la información de CONTEXTO no es suficiente, responde: "Lo siento, la información para responder a esta pregunta no se encuentra en el material de estudio que tengo indexado."
        Tu respuesta debe ser concisa y académica.
        """

        prompt = f"""
        CONTEXTO:
        ---
        {context_str}
        ---

        PREGUNTA DEL USUARIO: {request.pregunta}
        """

        # 4. Llamar a Mistral 7B (32K tokens) para generar la respuesta
        ollama_response = client.chat(
            model=GENERATION_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
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
    setup_database()
    print(f"Iniciando Athanor PoC con {GENERATION_MODEL} en http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    setup_database()