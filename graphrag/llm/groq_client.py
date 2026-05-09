import json
import time
from typing import List, Dict, Any, Type, TypeVar, Optional

from groq import Groq
from pydantic import BaseModel

# Importamos la configuración. Ajusta la ruta (..) según la estructura de tus carpetas
from ..config import get_groq_settings 

T = TypeVar('T', bound=BaseModel)

class GroqClient:
    def __init__(self):
        # Obtenemos los settings de Pydantic
        self.settings = get_groq_settings()
        
        # Instanciamos el cliente oficial usando la key del .env
        self.client = Groq(
            api_key=self.settings.groq_api_key,
            # Groq acepta parámetros adicionales como timeout si los necesitas
            timeout=60.0
        )

    def _parse_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Groq utiliza el formato estándar de OpenAI (lista de dicts con role y content).
        """
        return messages

    def _rate_limit_delay(self):
        """Pausa para evitar límites de tasa de Groq."""
        time.sleep(2)

    def chat(
            self,
            messages: List[Dict[str, str]],
            model: str = None,
            temperature: float = 0.0,
            format: str = None
    ) -> str:
        """Genera una respuesta de texto estándar o JSON libre."""
        # Usa el modelo pasado por parámetro o el configurado en settings
        model_name = model or self.settings.groq_model
        parsed_messages = self._parse_messages(messages)

        params = {
            "model": model_name,
            "messages": parsed_messages,
            "temperature": temperature,
        }

        if format == "json":
            params["response_format"] = {"type": "json_object"}

        self._rate_limit_delay()

        completion = self.client.chat.completions.create(**params)
        return completion.choices[0].message.content

    def structured_output(
            self,
            prompt: str,
            schema: Type[T],
            system_prompt: str = "You are an expert data extractor. Respond strictly in JSON format.",
            model: str = None
    ) -> T:
        """Fuerza al modelo a devolver un objeto basado en un esquema de Pydantic."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.structured_output_with_chat(messages, schema, model)

    def structured_output_with_chat(
            self,
            messages: List[Dict[str, str]],
            schema: Type[T],
            model: str = None,
            temperature: float = 0.0
    ) -> T:
        """Genera output estructurado usando JSON mode con inyección de esquema."""
        model_name = model or self.settings.groq_model
        
        enriched_messages = messages.copy()
        schema_instruction = f"\nReturn a JSON object that matches this schema: {schema.model_json_schema()}"
        
        if enriched_messages[0]["role"] == "system":
            enriched_messages[0]["content"] += schema_instruction
        else:
            enriched_messages.insert(0, {"role": "system", "content": f"You are a helpful assistant. {schema_instruction}"})

        self._rate_limit_delay()

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=enriched_messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        return schema.model_validate_json(completion.choices[0].message.content)

    def embed(self, texts: List[str], model: str = "nomic-embed-text-v1.5") -> List[List[float]]:
        raise NotImplementedError("Groq se especializa en inferencia de Chat. Para embeddings, te recomiendo usar la clase Ollama u otro proveedor.")

    @staticmethod
    def extract_json(response: str) -> Dict[str, Any]:
        """Extrae JSON de la respuesta."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start = response.find('```json')
            if start != -1:
                start = response.find('\n', start) + 1
                end = response.find('```', start)
                if end != -1:
                    return json.loads(response[start:end].strip())
            
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end + 1])
            raise ValueError("No se pudo extraer JSON de la respuesta")