import requests
import json

print("🔗 Тестируем Ollama...")

try:
    response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={
            "model": "deepseek-r1:8b", 
            "prompt": "Привет! Ответь кратмо на русском. Сколько человек нужно для игры в гандбол?",
            "stream": True,
        },
        timeout=30,
        stream=True  # ← ВАЖНО: stream=True для requests
    )
    
    if response.status_code == 200:
        print("✅ Успех! Ответ:")
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    full_response += data["response"]
        print(full_response)
    else:
        print(f"❌ Ошибка: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Ошибка подключения: {e}")
    print("Проверь что:")
    print("1. Ollama запущен на хосте") 
    print("2. Модель deepseek-r1:8b скачана: ollama pull deepseek-r1:8b")