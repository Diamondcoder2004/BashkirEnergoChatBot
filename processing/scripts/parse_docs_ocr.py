# main.py — PDF → Markdown с OCR и семантическим чанкингом 
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime

# Пути
INPUT_DIR = Path("/app/data/documents")  # было /app/documents
OUTPUT_DIR = Path("/app/data/output")    # было /app/output
OUTPUT_DIR.mkdir(exist_ok=True)

# Типы документов и ключевые слова
DOCUMENT_TYPES = {
    "закон": ["федеральный закон", "закон", "постановление", "правила"],
    "договор": ["договор", "соглашение", "контракт"],
    "акт": ["акт", "накладная", "счёт-фактура", "реестр"],
    "заявление": ["заявление", "ходатайство", "запрос", "обращение"],
    "инструкция": ["инструкция", "регламент", "методические указания", "руководство"],
    "приказ": ["приказ", "распоряжение"],
    "прочее": []
}

# Теги по ключевым словам
TAGS_KEYWORDS = {
    "конфиденциально": ["конфиденци", "ограниченный доступ", "служебная тайна"],
    "подписано": ["подпись", "расшифровка", "М.П.", "эцп", "удостоверяю"],
    "с печатью": ["печать", "штамп", "оттиск"],
    "энергетика": ["электроэнергия", "тариф", "подключение", "сети", "мощность"],
    "финансы": ["оплата", "платёж", "счёт", "реквизиты", "банк"],
    "юридический": ["юридический", "адрес", "ОГРН", "ИНН", "КПП"]
}

def clean_author_name(author: str) -> str:
    """Очищает имя автора от специальных символов"""
    if not author or author == "Не указано":
        return "Не указано"
    
    # Удаляем не-печатные символы
    cleaned = re.sub(r'[^\x20-\x7E\u0400-\u04FF]', '', author)
    
    # Если после очистки строка пустая или слишком короткая
    if not cleaned.strip() or len(cleaned.strip()) < 2:
        return "Не указано"
    
    return cleaned.strip()

def classify_document(text_sample: str) -> str:
    """Быстрая классификация документа"""
    text_lower = text_sample.lower()
    for doc_type, keywords in DOCUMENT_TYPES.items():
        if any(kw in text_lower for kw in keywords):
            return doc_type
    return "прочее"

def generate_tags(text_sample: str) -> list:
    """Быстрая генерация тегов"""
    text_lower = text_sample.lower()
    tags = []
    for tag, keywords in TAGS_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    return sorted(set(tags)) if tags else ["без тегов"]

def get_pdf_metadata(pdf_path: Path):
    """Быстрое получение метаданных PDF"""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata
        author = meta.get("author", "Не указано").strip() or "Не указано"
        title = meta.get("title", pdf_path.stem).strip() or pdf_path.stem
        created = meta.get("creationDate", "")
        created = created.replace("D:", "")[:10] if created else "Не указано"
        pages = len(doc)
        doc.close()
        return author, title, created, pages
    except Exception as e:
        print(f"⚠️  Не удалось прочитать метаданные: {e}")
        return "Не указано", pdf_path.stem, "Не указано", "?"

def is_scanned_pdf(pdf_path: Path) -> bool:
    """Проверяем, является ли PDF сканированным изображением"""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        has_text = False
        
        # Проверяем первые 3 страницы
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text().strip()
            if text and len(text) > 50:  # Если есть значительный текст
                has_text = True
                break
        
        doc.close()
        return not has_text  # Если текста нет - значит сканированный
        
    except Exception as e:
        print(f"⚠️  Ошибка проверки типа PDF: {e}")
        return True  # В случае ошибки считаем сканированным

def extract_text_with_ocr(pdf_path: Path) -> str:
    """Извлечение текста через OCR для сканированных PDF"""
    try:
        from pdf2image import convert_from_path
        import pytesseract
        from PIL import Image
        
        print(f"🔍 OCR обработка: {pdf_path.name}")
        
        # Конвертируем PDF в изображения
        images = convert_from_path(
            str(pdf_path),
            dpi=300,  # Высокое качество для лучшего OCR
            poppler_path='/usr/bin'  # Путь к poppler в Docker
        )
        
        all_text = []
        
        for i, image in enumerate(images):
            print(f"  📄 Страница {i+1}/{len(images)}")
            
            # Увеличиваем контраст для лучшего распознавания
            image = image.convert('L')  # В grayscale
            
            # Используем tesseract с русским и английским языками
            text = pytesseract.image_to_string(
                image, 
                lang='rus+eng',
                config='--psm 3 --oem 3'
            )
            
            if text.strip():
                all_text.append(f"## Страница {i+1}\n\n{text}\n\n")
        
        return "\n".join(all_text) if all_text else None
        
    except Exception as e:
        print(f"❌ OCR ошибка: {e}")
        return None

def extract_text_with_pymupdf(pdf_path: Path) -> str:
    """Извлечение текста из обычного PDF"""
    try:
        import fitz
        
        doc = fitz.open(str(pdf_path))
        markdown_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                clean_text = re.sub(r'\n{3,}', '\n\n', text)
                markdown_parts.append(f"## Страница {page_num + 1}\n\n{clean_text}\n\n")
        
        doc.close()
        return "\n".join(markdown_parts) if markdown_parts else None
        
    except Exception as e:
        print(f"❌ PyMuPDF ошибка: {e}")
        return None

def convert_pdf_to_markdown(pdf_path: Path):
    """Умная конвертация с автоматическим определением типа PDF"""
    print(f"📄 Обработка: {pdf_path.name}")
    
    # Сначала пробуем извлечь текст обычным способом
    full_text = extract_text_with_pymupdf(pdf_path)
    
    # Если текста мало или нет - используем OCR
    if not full_text or not full_text.strip() or len(full_text.strip()) < 100:
        print(f"🔄 Мало текста, применяем OCR: {pdf_path.name}")
        ocr_text = extract_text_with_ocr(pdf_path)
        if ocr_text:
            full_text = ocr_text
            print(f"✅ OCR успешно: {pdf_path.name}")
        else:
            print(f"❌ OCR не дал результатов: {pdf_path.name}")
    
    # Если оба метода не сработали
    if not full_text or not full_text.strip():
        print(f"⚠️  Не удалось извлечь текст: {pdf_path.name}")
        return
    
    # Образец текста для анализа
    text_sample = full_text[:2000] if len(full_text) > 2000 else full_text
    
    # Получаем метаданные
    author, title, created, pages = get_pdf_metadata(pdf_path)
    # ОЧИЩАЕМ автора из-за 6 документа
    author = clean_author_name(author)
    # Генерируем метаданные (БЕЗ SUMMARY)
    doc_type = classify_document(text_sample)
    tags = generate_tags(text_sample)
    
    # YAML-метаданные (убрали summary)
    yaml_header = f"""---
filename: {pdf_path.name}
title: {title}
author: {author}
creation_date: {created}
pages: {pages}
document_type: {doc_type}
tags: {json.dumps(tags, ensure_ascii=False)}
processed_at: {datetime.now().isoformat().split('.')[0]}
---

"""
    # Сохраняем результат
    output_path = OUTPUT_DIR / f"{pdf_path.stem}.md"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_header.strip() + "\n\n" + full_text.strip())
        print(f"✅ Успешно: {pdf_path.name}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {pdf_path.name} - {e}")

def main():
    print(f"🔍 Поиск PDF в {INPUT_DIR}")
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"📚 Найдено {len(pdf_files)} документов")
    
    if not pdf_files:
        print("❌ PDF файлы не найдены")
        return
    
    total_start = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] " + "="*50)
        
        doc_start = time.time()
        convert_pdf_to_markdown(pdf_path)
        doc_time = time.time() - doc_start
        
        if doc_time > 120:
            print(f"⚠️  Документ обрабатывался долго: {doc_time:.1f} сек")
        
        # Пауза между документами
        if i < len(pdf_files):
            time.sleep(1)
    
    total_time = time.time() - total_start
    print(f"\n🎉 Все документы обработаны за {total_time:.1f} секунд")
    print(f"📁 Результат: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()