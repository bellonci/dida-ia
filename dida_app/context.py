"""Conservative context budgets and query-ranked document excerpts.

UTF-8 bytes are an upper-bound estimate for common byte-BPE text tokenizers,
not the provider's exact tokenizer. Image cost is an explicit reservation.
"""
import copy
import re
import unicodedata

IMAGE_RESERVE = 4096


def text_tokens(text):
    return len(text.encode('utf-8'))


def clip_text(text, budget):
    if budget <= 0:
        return ''
    return text.encode('utf-8')[:budget].decode('utf-8', errors='ignore')


def message_tokens(message):
    content = message.get('content', '')
    if isinstance(content, str):
        return 16 + text_tokens(content)
    return 16 + sum(IMAGE_RESERVE if part.get('type') == 'image_url'
                    else text_tokens(part.get('text', '')) + 8 for part in content)


def normalize(text):
    return ''.join(char for char in unicodedata.normalize('NFKD', text.casefold())
                   if not unicodedata.combining(char))


def document_excerpt(text, query, budget=3500):
    if text_tokens(text) <= budget:
        return text
    chunks = [(offset, text[offset:offset + 1200]) for offset in range(0, len(text), 1050)]
    words = {word for word in re.findall(r'\w+', normalize(query)) if len(word) >= 4}
    ranked = sorted(chunks, key=lambda pair: (-sum(normalize(pair[1]).count(word) for word in words), pair[0]))
    if not words or not any(any(word in normalize(chunk) for word in words) for _, chunk in chunks):
        # With no topic-specific match, sample beginning, middle and end.
        indices = list(dict.fromkeys([0, len(chunks)//2, len(chunks)-1]))
        ranked = [chunks[index] for index in indices] + [chunk for index, chunk in enumerate(chunks) if index not in indices]
    selected = []
    marker = '[Trechos selecionados; documento não enviado na íntegra.]\n'
    left = max(0, budget - text_tokens(marker))
    for offset, chunk in ranked:
        label = f'\n[Trecho a partir do caractere {offset}]\n'
        remaining = left - text_tokens(label)
        if remaining < 80:
            break
        value = clip_text(chunk, remaining)
        selected.append((offset, label + value))
        left -= text_tokens(label + value)
    return clip_text(marker + ''.join(value for _, value in sorted(selected)), budget)


def attachment_excerpt(attachment, query, budget=3500):
    documents = attachment.get('documents')
    if not documents:
        return document_excerpt(attachment.get('text', ''), query, budget)
    # Fair per-document allocation retains every selected filename.
    pieces = []
    per_document = max(1, budget // len(documents))
    for document in documents:
        name = str(document.get('filename', 'arquivo'))
        label = f'\nArquivo: {name}\n'
        pieces.append(clip_text(label + document_excerpt(document.get('content_excerpt', ''), query,
                           max(0, per_document - text_tokens(label))), per_document))
    return clip_text(''.join(pieces), budget)


def limit_messages(messages, budget=8192):
    if budget < 64:
        raise ValueError('Orçamento de contexto insuficiente')
    result = copy.deepcopy(messages)
    if not result:
        return result
    # Keep system instructions and the current question; drop older history first.
    while len(result) > 2 and sum(message_tokens(message) for message in result) > budget:
        result.pop(1)
    if len(result) == 1:
        result[0]['content'] = clip_text(result[0]['content'], budget - 16)
        return result
    current = result[-1]
    content = current['content']
    image_cost = 0
    if isinstance(content, list):
        image_cost = sum(IMAGE_RESERVE if part.get('type') == 'image_url' else 8 for part in content)
    if image_cost + 64 > budget:
        raise ValueError('O modelo não tem contexto suficiente para esta imagem')
    system_budget = min(text_tokens(result[0]['content']), (budget - image_cost - 32)//2)
    result[0]['content'] = clip_text(result[0]['content'], system_budget)
    remaining = budget - message_tokens(result[0]) - image_cost - 16
    if isinstance(content, str):
        current['content'] = clip_text(content, remaining)
    else:
        for part in content:
            if part.get('type') == 'text':
                part['text'] = clip_text(part['text'], remaining)
                remaining -= text_tokens(part['text'])
    assert sum(message_tokens(message) for message in result) <= budget
    return result
