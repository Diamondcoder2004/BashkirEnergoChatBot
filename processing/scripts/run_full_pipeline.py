#!/usr/bin/env python3
"""
Консольное приложение для управления RAG-конвейером
"""
import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем настройки
load_dotenv('/app/.env')

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'bashkir_energo_docs')

def run_command(cmd, description):
    """Выполняет команду и выводит результат"""
    print(f"\n🚀 {description}")
    print(f"Команда: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(cmd, shell=isinstance(cmd, str), 
                              check=True, capture_output=True, text=True)
        print("✅ Успешно выполнено")
        if result.stdout:
            print(f"Вывод: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        if e.stderr:
            print(f"Вывод ошибки: {e.stderr}")
        return False

def parse_pdf_to_md():
    """1. Парсинг PDF в Markdown"""
    print("\n" + "="*60)
    print("📄 Парсинг PDF в Markdown с OCR")
    return run_command(["python", "scripts/parse_docs_ocr.py"], "Парсинг PDF документов")

def semantic_chunking():
    """2. Семантический чанкинг"""
    print("\n" + "="*60)
    print("🔪 Семантический чанкинг документов")
    return run_command(["python", "scripts/semantic_chunking.py"], "Семантический чанкинг")

def chunk_encoding():
    """3. Кодирование чанков в векторную БД"""
    print("\n" + "="*60)
    print("🗄️ Кодирование чанков в векторную БД")
    return run_command(["python", "scripts/chunk_embedding_qdrant.py"], "Создание векторной базы")

def change_encoder_qdrant():
    """4. Смена энкодера в Qdrant"""
    print("\n" + "="*60)
    print("🔄 Смена энкодера в Qdrant")
    return run_command(["python", "scripts/change_embedder_qdrant.py"], "Переиндексация с rubert моделью")

def check_dependencies():
    """5. Проверка зависимостей"""
    print("\n" + "="*60)
    print("🔍 Проверка зависимостей")
    
    try:
        import requests
        
        # Проверяем Qdrant
        try:
            response = requests.get(f"http://{os.getenv('QDRANT_HOST', 'localhost')}:6333", timeout=5)
            print("✅ Qdrant: доступен")
        except:
            print("❌ Qdrant: недоступен")
        
        # Проверяем Ollama
        try:
            response = requests.get(f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/tags", timeout=5)
            print("✅ Ollama: доступен")
        except:
            print("⚠️  Ollama: недоступен")
            
        # Проверяем директории
        dirs_to_check = [
            ("📁 Исходные документы", Path("/app/data/documents")),
            ("📁 Обработанные MD", Path("/app/data/output")),
            ("📁 Семантические чанки", Path("/app/data/semantic_chunks"))
        ]
        
        for name, path in dirs_to_check:
            if path.exists():
                files = list(path.glob("*"))
                print(f"{name}: {len(files)} файлов")
            else:
                print(f"{name}: директория не существует")
                
        print(f"\n🔧 Текущие настройки:")
        print(f"   Модель эмбеддингов: {EMBEDDING_MODEL}")
        print(f"   Коллекция: {COLLECTION_NAME}")
        
    except ImportError:
        print("❌ requests не установлен")

def main():
    """Основное меню"""
    print("🤖 RAG Pipeline Manager")
    
    # Создаем необходимые директории
    Path("/app/data/documents").mkdir(parents=True, exist_ok=True)
    Path("/app/data/output").mkdir(parents=True, exist_ok=True)
    Path("/app/data/semantic_chunks").mkdir(parents=True, exist_ok=True)
    
    while True:
        print("\n" + "="*40)
        print("Выберите команду:")
        print("1. parse_pdf_to_md")
        print("2. semantic_chunking") 
        print("3. chunk_encoding")
        print("4. change_encoder_qdrant")
        print("5. check_dependencies")
        print("6. exit")
        print("-"*40)
        
        choice = input("Введите номер команды (1-6): ").strip()
        
        if choice == '1':
            parse_pdf_to_md()
        elif choice == '2':
            semantic_chunking()
        elif choice == '3':
            chunk_encoding()
        elif choice == '4':
            change_encoder_qdrant()
        elif choice == '5':
            check_dependencies()
        elif choice == '6':
            print("👋 Выход...")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()