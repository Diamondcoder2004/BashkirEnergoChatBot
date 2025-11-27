#!/usr/bin/env python3
"""
Скрипт для запуска полного процесса RAG:
1. OCR и парсинг PDF документов
2. Семантический чанкинг
3. Создание векторной базы в Qdrant
"""
import subprocess
import sys
import os
from pathlib import Path


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
        print(f"Вывод ошибки: {e.stderr}")
        return False


def main():
    print("🤖 Запуск полного RAG-конвейера")
    
    # Проверяем существование директорий
    Path("./documents").mkdir(exist_ok=True)
    Path("./output").mkdir(exist_ok=True)
    Path("./semantic_chunks").mkdir(exist_ok=True)
    
    # Шаг 1: OCR и парсинг PDF
    print("\n" + "="*60)
    success = run_command([
        "python", "scripts/parse_docs_ocr.py"
    ], "Шаг 1: OCR и парсинг PDF документов")
    
    if not success:
        print("❌ Ошибка на шаге 1. Завершение.")
        sys.exit(1)
    
    # Шаг 2: Семантический чанкинг
    print("\n" + "="*60)
    success = run_command([
        "python", "scripts/semantic_chunking.py"
    ], "Шаг 2: Семантический чанкинг документов")
    
    if not success:
        print("❌ Ошибка на шаге 2. Завершение.")
        sys.exit(1)
    
    # Шаг 3: Создание векторной базы
    print("\n" + "="*60)
    success = run_command([
        "python", "scripts/ai_chunking.py"
    ], "Шаг 3: Создание векторной базы в Qdrant")
    
    if not success:
        print("❌ Ошибка на шаге 3. Завершение.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 Полный RAG-конвейер успешно завершен!")
    print("📊 Векторная база создана в Qdrant")
    print("🔍 Коллекция: bashkir_energo_docs_nomic")
    print("🌐 Qdrant Dashboard: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()