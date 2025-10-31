Entiendo tu intención, ¡pero **no puedo acceder directamente a tu Google Drive** ni interactuar con servicios externos como Google Docs por ti\! 😔

Soy un modelo de lenguaje y no tengo permisos de acceso a sistemas de archivos privados o servicios de usuario, como tu Drive.

-----

## 📄 ¿Cómo transferir la documentación de Athanor?

La mejor manera de llevar esta documentación a Google Docs es mediante la función de **copiar y pegar el texto en formato Markdown**. Google Docs es muy bueno para interpretar el formato Markdown (los encabezados `##`, las listas, y las tablas) y darle el formato adecuado automáticamente.

Aquí tienes los pasos para generar la documentación final del proyecto:

1.  **Copia el Bloque Completo:** Copia el texto del último mensaje que resume el proyecto (incluyendo las secciones y bloques de código).
2.  **Abre Google Docs:** Crea un nuevo documento en tu Google Drive.
3.  **Pega el Contenido:** Pega el texto copiado.

-----

## 🛠️ Generación de la Documentación

Aquí tienes el contenido completo, final y corregido, que integra el nombre **Athanor**, la estrategia de **Mistral 7B** y la **arquitectura PoC**.

-----

# 📜 Documentación del PoC: Tutor Athanor (v1.0)

Este documento sirve como la documentación inicial del proyecto **Athanor** (*El Horno Alquímico*), que implementa una arquitectura RAG para crear un tutor virtual especializado.

## 1\. 💡 Arquitectura y Estrategia

El PoC está diseñado para ser **100% On-Premise** y desacoplado, lo que permite la futura migración a un modelo híbrido. Se ejecuta sobre una única GPU **RTX 3080**.

### Modelos y Componentes Clave

| Componente | Modelo/Herramienta | VRAM Requerida (Aprox.) | Función Estratégica |
| :--- | :--- | :--- | :--- |
| **Generación (LLM)** | **Mistral 7B** (`mistral:7b`) | $\sim 8$ GB | Elegido por su **ventana de contexto de $32,768$ tokens**, esencial para la gestión de diálogos largos y la memoria conversacional. |
| **Embeddings (RAG)** | **BGE-M3** (`bge-m3`) | $\sim 2$ GB | Modelo de *embedding* eficiente y *open-source* (1024 dimensiones) para la Ingesta de documentos. |
| **Base de Datos** | **PostgreSQL + `pgvector`** | Mínima | Almacenamiento persistente de los vectores (memoria RAG) y metadatos. |

### Flujo RAG (Ingesta y Generación)

1.  **Ingesta (Fábrica):** El *endpoint* `/upload` recibe un PDF, lo fragmenta (`nltk.sent_tokenize`), lo vectoriza usando **BGE-M3**, y almacena el `chunk` y el `embedding` en **`pgvector`**.
2.  **Generación (Inferencia):** El *endpoint* `/generate` usa **BGE-M3** para vectorizar la pregunta del usuario, consulta **`pgvector`** para recuperar los *Top-K* fragmentos relevantes, y utiliza el **System Prompt** para forzar a **Mistral 7B** a actuar como un tutor.

-----

## 2\. 🐳 Configuración de Despliegue (VM 1 - GPU)

El proyecto se despliega utilizando `docker-compose` en la VM con aceleración NVIDIA.

### A. `ai-backend/docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      # Las APIs se comunican internamente por nombres de servicio (ollama, db)
      - DB_HOST=db
      - DB_USER=admin
      - DB_PASS=admin
      - DB_NAME=rag_poc
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - db
      - ollama
    restart: always

  db:
    image: ankane/pgvector
    environment:
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=admin
      - POSTGRES_DB=rag_poc
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: always

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: always

volumes:
  pgdata:
  ollama-data:
```

### B. `ai-backend/api/requirements.txt`

```
fastapi[all]
uvicorn
pypdf
psycopg2-binary
ollama
nltk
```

### C. `ai-backend/api/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

-----

## 3\. 🐍 `ai-backend/api/main.py` (Versión Tutor Final)

(Este archivo contiene la lógica de conexión, la creación de tablas, el *chunking* y el rol del tutor).

```python
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

# Configuración y Modelos (Mistral 7B) ...
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASS = os.getenv("DB_PASS", "admin")
DB_NAME = os.getenv("DB_NAME", "rag_poc")
DB_PARAMS = {"host": DB_HOST, "port": 5432, "user": DB_USER, "password": DB_PASS, "database": DB_NAME}
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "mistral:7b" 
VECTOR_DIMENSION = 1024 
client = ollama.Client(host=OLLAMA_HOST)
nltk.download('punkt', quiet=True)
app = FastAPI(title="Athanor AI Backend")

@contextmanager
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        yield conn
    except Exception as e:
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
                id SERIAL PRIMARY KEY, filename TEXT NOT NULL,
                chunk TEXT NOT NULL, embedding VECTOR({VECTOR_DIMENSION})
            );
            """)
            conn.commit()

# --- ENDPOINT DE INGESTA (/upload) ---
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")
    try:
        pdf_bytes = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        chunks = nltk.sent_tokenize(full_text)
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    if not chunk.strip(): continue
                    response = client.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
                    embedding = response["embedding"]
                    cur.execute(
                        "INSERT INTO documents (filename, chunk, embedding) VALUES (%s, %s, %s)",
                        (file.filename, chunk, embedding)
                    )
                conn.commit()

        return {"status": "success", "filename": file.filename, "chunks_stored": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {e}")

# --- ENDPOINT DE GENERACIÓN (/generate) ---
class QueryRequest(BaseModel):
    pregunta: str
    top_k: int = 5

@app.post("/generate")
async def generate_response(request: QueryRequest):
    try:
        response = client.embeddings(model=EMBEDDING_MODEL, prompt=request.pregunta)
        query_embedding = response["embedding"]

        context_chunks = []
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk FROM documents ORDER BY embedding <-> %s::vector LIMIT %s",
                    (str(query_embedding), request.top_k)
                )
                context_chunks = [row[0] for row in cur.fetchall()]
        
        if not context_chunks:
            return {"respuesta": "Lo siento, la información para responder a esta pregunta no se encuentra en el material de estudio que tengo indexado.", "contexto": []}

        # SYSTEM PROMPT 'Athanor Tutor'
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
        raise HTTPException(status_code=500, detail=f"Error en la generación: {e}")

if __name__ == "__main__":
    setup_database()
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    setup_database()
```