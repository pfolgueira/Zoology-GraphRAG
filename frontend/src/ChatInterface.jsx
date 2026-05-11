import React, { useState, useRef, useEffect } from 'react';

// Sub-componente para renderizar cada mensaje
const MessageBubble = ({ message }) => {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex w-full mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div 
        className={`max-w-[75%] px-4 py-3 rounded-2xl shadow-sm ${
          isUser 
            ? 'bg-[#2FA084] text-white rounded-br-none' 
            : 'bg-[#EEEEEE] text-gray-800 rounded-bl-none border border-gray-200'
        }`}
      >
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
          {message.content}
        </p>
      </div>
    </div>
  );
};

export default function ChatInterface() {
  const initialMessage = { role: 'assistant', content: '¡Hola! Soy tu Asistente Virtual. ¿En qué puedo ayudarte hoy?' };
  
  const [messages, setMessages] = useState([initialMessage]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [requestTimestamps, setRequestTimestamps] = useState([]);
  const [rateLimitError, setRateLimitError] = useState(false);
  
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-ajustar la altura del textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  // Auto-scroll al final cuando hay nuevos mensajes o cambia el estado de carga
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, rateLimitError]);

  // Limpiar el mensaje de error de rate limit después de un tiempo
  useEffect(() => {
    if (rateLimitError) {
      const timer = setTimeout(() => setRateLimitError(false), 5000);
      return () => clearTimeout(timer);
    }
  }, [rateLimitError]);

// --- FUNCIÓN ACTUALIZADA: REINICIAR CONVERSACIÓN ---
  const handleReset = async () => {
    // 1. Limpiar estado local primero para que se sienta rápido
    setMessages([initialMessage]);
    setInputValue('');
    setIsLoading(false);
    setRequestTimestamps([]);
    setRateLimitError(false);

    // 2. Avisar al backend de Python que limpie la memoria de AgenticRAG
    try {
      await fetch('http://localhost:8000/api/reset', {
        method: 'POST',
      });
    } catch (error) {
      console.error("Error al reiniciar la memoria en el servidor:", error);
    }
  };

  // --- FUNCIÓN ACTUALIZADA: ENVIAR MENSAJE ---
  const handleSendMessage = async () => {
    const trimmedInput = inputValue.trim();
    if (!trimmedInput || isLoading) return;

    // --- Control de límite de tasa en Frontend ---
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const recentRequests = requestTimestamps.filter(timestamp => timestamp > oneMinuteAgo);
    
    if (recentRequests.length >= 15) {
      setRateLimitError(true);
      return; 
    }
    
    setRequestTimestamps([...recentRequests, now]);
    // -------------------------------------------------------------

    // Actualizar UI con el mensaje del usuario
    setMessages(prev => [...prev, { role: 'user', content: trimmedInput }]);
    setInputValue('');
    setIsLoading(true);
    setRateLimitError(false);

    // --- LLAMADA REAL AL BACKEND (FASTAPI) ---
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: trimmedInput }),
      });

      if (!response.ok) {
        // Manejar el error 429 de Rate Limit del backend
        if (response.status === 429) {
          setRateLimitError(true);
          setIsLoading(false);
          return;
        }
        throw new Error(`Error HTTP: ${response.status}`);
      }

      const data = await response.json();
      
      // Agregar la respuesta de tu AgenticRAG a la interfaz
      setMessages(prev => [
        ...prev, 
        { role: 'assistant', content: data.answer }
      ]);
      
    } catch (error) {
      console.error("Error de conexión:", error);
      // Mostrar un mensaje de error como si fuera el asistente para no romper el flujo
      setMessages(prev => [
        ...prev, 
        { role: 'assistant', content: "Hubo un problema de conexión con el servidor. Asegúrate de que el backend está corriendo." }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Evitar salto de línea si no se usa Shift
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen w-full bg-white overflow-hidden">
      {/* Cabecera */}
      <header className="flex justify-between items-center px-6 py-4 bg-white border-b border-gray-200 shrink-0 z-10">
        <h1 className="text-xl font-bold text-gray-800 tracking-tight">
          Asistente Virtual
        </h1>
        <button 
          onClick={handleReset}
          className="px-4 py-2 text-sm font-medium text-gray-600 bg-transparent border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-200"
        >
          Reiniciar Conversación
        </button>
      </header>

      {/* Zona de Chat */}
      <main className="flex-1 overflow-y-auto w-full p-4 sm:p-6 bg-gray-50/50">
        <div className="max-w-4xl mx-auto w-full">
          {messages.map((msg, index) => (
            <MessageBubble key={index} message={msg} />
          ))}

          {/* Estado de carga del asistente */}
          {isLoading && (
            <div className="flex w-full mb-4 justify-start">
              <div className="px-4 py-4 bg-[#EEEEEE] rounded-2xl rounded-bl-none border border-gray-200 flex items-center gap-1.5">
                <div className="w-2 h-2 bg-[#2FA084] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-[#2FA084] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-[#2FA084] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          )}

          {/* Mensaje de error de Rate Limit */}
          {rateLimitError && (
            <div className="flex justify-center mb-4">
              <div className="bg-red-50 text-red-600 px-4 py-2 rounded-lg text-sm font-medium shadow-sm border border-red-100">
                Has superado el límite de peticiones (15/minuto). Por favor, espera un momento.
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Zona de Entrada */}
      <footer className="shrink-0 p-4 bg-white">
        <div className="max-w-4xl mx-auto w-full relative flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="Escribe tu mensaje aquí..."
            className="flex-1 min-h-[52px] resize-none overflow-hidden rounded-xl border border-gray-300 bg-white py-3 px-4 text-gray-800 shadow-sm focus:border-[#2FA084] focus:outline-none focus:ring-1 focus:ring-[#2FA084] disabled:bg-[#EEEEEE] disabled:text-gray-500"
            rows={1}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="flex shrink-0 items-center justify-center p-3 h-[52px] w-[52px] bg-[#2FA084] text-white rounded-xl shadow-sm hover:bg-[#1F6F5F] transition-colors disabled:bg-[#6FCF97] disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-[#2FA084] focus:ring-offset-2"
            aria-label="Enviar mensaje"
          >
            <svg 
              xmlns="http://www.w3.org/2000/svg" 
              viewBox="0 0 24 24" 
              fill="currentColor" 
              className="w-5 h-5 ml-1"
            >
              <path d="M3.478 2.404a.75.75 0 00-.926.941l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.404z" />
            </svg>
          </button>
        </div>
        <p className="text-center text-xs text-gray-400 mt-2">
          Presiona Enter para enviar, Shift + Enter para saltar de línea.
        </p>
      </footer>
    </div>
  );
}