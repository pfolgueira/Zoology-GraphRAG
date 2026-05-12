# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE TU SISTEMA RAG ---
sys.path.append('.')
load_dotenv("./env/.env")

from graphrag.graph.neo4j_manager import Neo4jManager
from graphrag.agents import AgenticRAG

# Inicialización de Neo4j y RAG
neo4j = Neo4jManager()
agentic_rag = AgenticRAG(neo4j)

# Reiniciar historial al iniciar el servidor
agentic_rag.reset_conversation()

# --- CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(title="GraphRAG API")

# IMPORTANTE: Configurar CORS para permitir peticiones desde React (Vite/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia esto por la URL de tu frontend, ej: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Llamamos a tu sistema RAG
        result = agentic_rag.answer(request.message)
        return ChatResponse(answer=result['answer'])
    except Exception as e:
        error_msg = str(e).lower()
        # Capturamos el error de Rate Limit de la API (15 peticiones/minuto)
        if "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg or "too many requests" in error_msg:
            raise HTTPException(status_code=429, detail="Límite de peticiones de la API alcanzado. Por favor, espera un minuto.")
        
        # Cualquier otro error del agente
        raise HTTPException(status_code=500, detail=f"Error interno del agente: {str(e)}")

@app.post("/api/reset")
async def reset_endpoint():
    try:
        agentic_rag.reset_conversation()
        return {"status": "Conversación reiniciada exitosamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al reiniciar la conversación")

if __name__ == "__main__":
    import uvicorn
    # Inicia el servidor en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)