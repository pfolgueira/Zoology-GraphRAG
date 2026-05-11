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

# Configuración de ejemplos y terminología
text2cypher_examples = [
    ("How many Mammal species are there?", "MATCH (s:Species)-[:BELONGS_TO_CLASS]->(c:AnimalClass {type: 'Mammal'}) RETURN count(s) AS totalMammals"),
    ("What is the top speed and maximum weight of a Lion?", "MATCH (s:Species {name: 'Lion'}) RETURN s.top_speed_kmh, s.weight_max_kg"),
    ("What type of diet do tigers have and what do they prey on?", "MATCH (s:Species {name: 'Tiger'})-[:HAS_DIET_TYPE]->(d:DietType), (s)-[:PREYS_ON]->(prey:Species) RETURN d.type, prey.name"),
    ("Which animals have a lifespan greater than 50 years?", "MATCH (s:Species) WHERE s.lifespan_years > 50 RETURN s.name, s.lifespan_years"),
    ("Where do whales migrate to in the winter?", "MATCH (s:Species {name: 'Whale'})-[m:MIGRATES_TO {season: 'winter'}]->(l:Location) RETURN l.type"),
    ("What does an chimpanzee eat?", "MATCH (s:Species {name: 'Chimpanzee'}) OPTIONAL MATCH (s)-[:PREYS_ON]->(prey:Species) OPTIONAL MATCH (s)-[:FEEDS_ON]->(food:FoodSource) RETURN s.name AS species, collect(DISTINCT prey.name) AS preys_on, collect(DISTINCT food.type) AS feeds_on"),
    ("In which countries or regions are kangaroos found?", "MATCH (s:Species {name: 'Kangaroo'})-[:FOUND_IN]->(l:Location) RETURN l.type"),
    ("What kind of ecosystem or biome does the polar bear inhabit?", "MATCH (s:Species {name: 'Polar Bear'})-[:INHABITS]->(h:Habitat) RETURN h.type")
]

terminology_maps = [
    ("animal, creature, species", "Refers to the node with label (:Species)"),
    ("what does X eat, what do X eat, diet of X, food of X. When asking what an animal eats, check BOTH relationships: ", "[:PREYS_ON]->(:Species) for animal prey AND [:FEEDS_ON]->(:FoodSource) for non-animal food sources. Use OPTIONAL MATCH for both and return all results."),
    ("country, continent, geographic region, area, located in", "Use the relationship [:FOUND_IN]->(:Location) to refer to specific geographical and political places."),
    ("biome, ecosystem, type of environment, terrain, inhabits", "Use the relationship [:INHABITS]->(:Habitat) to refer to the natural biome or habitat type.")    
]

for question, cypher in text2cypher_examples:
    agentic_rag.add_text2cypher_example(question, cypher)

for terms, explanation in terminology_maps:
    agentic_rag.add_terminology_map(terms, explanation)

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