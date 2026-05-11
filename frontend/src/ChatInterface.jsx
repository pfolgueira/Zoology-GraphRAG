import React, { useState, useRef, useEffect } from 'react';
import backgroundImg from './assets/01-background.png';

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
  const greetingOptions = [
    {
      text: "Greetings! I'm your Virtual Assistant. Don't mind the muscles; I only use them to carry the heavy weight of our zoology database. Ready to swing into some research, or should we go grab some bananas first?",
      emoji: "🦍"
    },
    {
      text: "Hi there! I was blending in with the background... did you see me? I'm an expert at camouflaging myself within thousands of documents to find exactly what you need. Ask away before I change colors again!",
      emoji: "🦎"
    },
    {
      text: "Hello! The view from up here is amazing—I can see the entire knowledge graph clearly. I've got the long neck needed to reach even the highest, most hidden data. What's sparking your curiosity today?",
      emoji: "🦒"
    },
    {
      text: "Welcome to the pride! I'm the one in charge of this data savanna. Don't worry, I've already had breakfast, so you can ask me anything without fear. Shall we start our expedition?",
      emoji: "🦁"
    },
    {
      text: "Splash! I'm your Virtual Assistant. I've just surfaced from the deep ocean of data to help you navigate. I'm a natural at finding 'current' information, so don't be shy—dive in with your questions!",
      emoji: "🐬"
    }
  ];
  
  const [messages, setMessages] = useState([]);
  const [greetingData, setGreetingData] = useState(greetingOptions[0]);
  
  useEffect(() => {
    setGreetingData(greetingOptions[Math.floor(Math.random() * greetingOptions.length)]);
  }, []);

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
    setMessages([]);
    setGreetingData(greetingOptions[Math.floor(Math.random() * greetingOptions.length)]);
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
        { role: 'assistant', content: "There was a connection problem with the server. Please make sure the backend is running." }
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

  const animalEmojis = ['🦁', '🐯', '🐼', '🐨', '🐸', '🐢', '🦖', '🐬', '🦘', '🦥', '🦩', '🦛'];
  const [animalEmoji, setAnimalEmoji] = useState('');

  useEffect(() => {
    const randomEmoji = animalEmojis[Math.floor(Math.random() * animalEmojis.length)];
    setAnimalEmoji(randomEmoji);
  }, []);

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden relative">
      {/* Imagen de fondo global */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <img src={backgroundImg} alt="background" className="w-full h-full object-cover" />
      </div>

      {/* Cabecera */}
      <header className="flex justify-center items-center px-6 py-4 bg-white/10 backdrop-blur-sm border-b border-gray-200 shrink-0 z-10 relative">
        <h1 className="text-3xl font-bold text-gray-800 tracking-tight flex items-center gap-2">
          {animalEmoji} Zoology Assistant
        </h1>
      </header>

      {/* Zona de Chat */}
      <main className="flex-1 overflow-y-auto w-full p-4 sm:p-6 relative z-10 bg-transparent flex flex-col">
        {messages.length === 0 && (
          <div className="flex-1 flex items-center justify-center p-4">
            <div className="max-w-2xl bg-white/80 backdrop-blur-md p-8 rounded-3xl shadow-lg border border-gray-200 text-center flex flex-col items-center">
              <span className="text-7xl mb-6">{greetingData.emoji}</span>
              <p className="text-lg md:text-xl text-gray-700 italic leading-relaxed font-medium">
                "{greetingData.text}"
              </p>
            </div>
          </div>
        )}
        
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
                You've exceeded the request limit (15/min). Please wait a moment.
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Zona de Entrada */}
      <footer className="shrink-0 p-4 bg-white/10 backdrop-blur-sm z-10 relative border-t border-gray-200/50">
        <div className="max-w-4xl mx-auto w-full relative flex items-end gap-2">
          {/* Reset button */}
          <button
            onClick={handleReset}
            title="Restart conversation"
            className="flex shrink-0 items-center justify-center p-3 h-[52px] w-[52px] sm:w-auto sm:px-4 bg-white text-gray-600 rounded-xl shadow-sm hover:bg-gray-50 border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#2FA084]"
            aria-label="Restart conversation"
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-5 h-5 sm:mr-2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>
            <span className="hidden sm:inline font-medium text-sm">Restart</span>
          </button>

          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="Type your message here..."
            className="flex-1 min-h-[52px] resize-none overflow-hidden rounded-xl border border-gray-300 bg-white py-3 px-4 text-gray-800 shadow-sm focus:border-[#2FA084] focus:outline-none focus:ring-1 focus:ring-[#2FA084] disabled:bg-[#EEEEEE] disabled:text-gray-500"
            rows={1}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="flex shrink-0 items-center justify-center p-3 h-[52px] w-[52px] bg-[#2FA084] text-white rounded-xl shadow-sm hover:bg-[#1F6F5F] transition-colors disabled:bg-[#6FCF97] disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-[#2FA084] focus:ring-offset-2"
            aria-label="Send message"
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
          Press Enter to send, Shift + Enter for newline.
        </p>
      </footer>
    </div>
  );
}