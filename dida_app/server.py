import json, threading, queue, base64
import secrets
from pathlib import Path
from http.server import BaseHTTPRequestHandler


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):

        def _trusted_request(self):
            from urllib.parse import urlsplit
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            host = self.headers.get('Host', '').lower()
            origin = self.headers.get('Origin')
            if host not in allowed:
                self.send_error(403, 'Host não autorizado')
                self.close_connection = True
                return False
            if origin is not None:
                try:
                    parsed = urlsplit(origin)
                except ValueError:
                    self.send_error(403, 'Origem inválida')
                    self.close_connection = True
                    return False
                if parsed.scheme != 'http' or parsed.netloc.lower() != host or parsed.path or parsed.query or parsed.fragment:
                    self.send_error(403, 'Origem não autorizada')
                    self.close_connection = True
                    return False
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send_error(403, 'Origem não autorizada')
                self.close_connection = True
                return False
            return True

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def _json(self, data, status=200):
            payload = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if not self._trusted_request():
                return
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(Path(__file__).with_name('ui.html').read_bytes())
            elif self.path == '/api/events':
                try:
                    client = app.estado.events.subscribe()
                except ValueError:
                    self.send_error(503, 'Muitas conexões')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                try:
                    while not client.closed.is_set():
                        try:
                            payload = client.receive(timeout=0.2)
                        except queue.Empty:
                            payload = {'type': 'level', 'levels': {'user': app.estado.level_user, 'bot': max(app.estado.level_dida, app.estado.fala.level)}, 'speaking': {'user': app.estado.speaking_user, 'bot': app.estado.speaking_dida or app.estado.fala.ocupada.is_set()}}
                        self.wfile.write(('data: ' + json.dumps(payload, ensure_ascii=False) + '\n\n').encode())
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    pass
                finally:
                    app.estado.events.unsubscribe(client)
            elif self.path.startswith('/api/files/'):
                try:
                    from urllib.parse import urlparse, parse_qs
                    path = urlparse(self.path).path
                    if path == '/api/files/status':
                        data = {'enabled': app.files_permission(), 'csrf': app.VOICE_CSRF}
                    elif path == '/api/files/job':
                        if not app.files_permission():
                            raise ValueError('Permissão revogada')
                        data = app.file_job_snapshot(parse_qs(urlparse(self.path).query).get('id', [''])[0])
                    else:
                        raise ValueError('Página não encontrada')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
                except (ValueError, OSError) as error:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(error)}, ensure_ascii=False).encode())
            elif self.path.startswith('/api/report?'):
                from urllib.parse import urlparse, parse_qs
                key = parse_qs(urlparse(self.path).query).get('id', [''])[0]
                with app.files_lock:
                    file = app.report_files.get(key)
                if file is None or not file.exists():
                    self.send_error(404)
                    return
                data = file.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/markdown; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="' + file.name + '"')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == '/api/voice':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'settings': app.voice_snapshot(), 'options': app.voice_catalog(), 'csrf': app.VOICE_CSRF}, ensure_ascii=False).encode())
            elif self.path.startswith('/api/sens'):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'v': app.estado.sensibilidade}).encode())
            elif self.path in ('/api/mem', '/api/mem/history'):
                data = {**app.memory.status(), 'csrf': app.VOICE_CSRF}
                if self.path == '/api/mem/history':
                    data['conversations'] = app.memory.history()
                self._json(data)
            elif self.path == '/api/models':
                self._json(app.model_catalog.status())
            elif self.path.startswith('/api/profile'):
                personal = app.profile_snapshot()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({**personal, 'greeting': f"Oi, {personal['name']}. Estou pronta para conversar."}, ensure_ascii=False).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self._trusted_request():
                return
            if self.headers.get('Transfer-Encoding'):
                self.send_error(400, 'Transfer-Encoding não suportado')
                self.close_connection = True
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                limit = app.MAX_FILE * 4 // 3 + 65536 if self.path == '/anexo' else 65536
                if length < 0 or length > limit:
                    self.send_error(413, 'Conteúdo grande demais')
                    self.close_connection = True
                    return
            except ValueError:
                self.send_error(400, 'Tamanho inválido')
                return
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                self.send_error(400, 'Corpo incompleto')
                return
            if self.path != '/anexo' or 'application/json' in self.headers.get('Content-Type', ''):
                try:
                    decoded = json.loads(raw_body.decode('utf-8'))
                    if not isinstance(decoded, dict):
                        raise ValueError('O pedido deve ser um objeto JSON')
                except (ValueError, UnicodeError):
                    self.send_error(400, 'JSON inválido')
                    return
            if self.path in ('/api/stop', '/api/mem/settings', '/api/mem/clear'):
                if not secrets.compare_digest(self.headers.get('X-Dida-CSRF', ''), app.VOICE_CSRF):
                    self.send_error(403, 'Reabra as configurações')
                    return
                try:
                    if self.path == '/api/stop':
                        app.cancel_response()
                    else:
                        with app.requests_manager.lock:
                            app.cancel_response()
                            if self.path == '/api/mem/settings':
                                app.memory.set_enabled(decoded.get('enabled'))
                            else:
                                app.memory.clear()
                    self._json({'ok': True, **app.memory.status()})
                except (ValueError, OSError) as error:
                    self._json({'error': str(error)}, 400)
                return
            if self.path.startswith('/api/files/'):
                if not secrets.compare_digest(self.headers.get('X-Dida-CSRF', ''), app.VOICE_CSRF):
                    self.send_error(403, 'Reabra o menu Arquivos')
                    self.close_connection = True
                    return
                try:
                    body = json.loads(raw_body.decode('utf-8'))
                    if not isinstance(body, dict):
                        raise ValueError('Pedido inválido')
                    if self.path == '/api/files/permission':
                        app.save_files_permission(body.get('enabled') is True)
                        data = {'enabled': app.files_permission()}
                    elif self.path == '/api/files/search':
                        key = app.start_file_search(body.get('query', ''), body.get('content') is True, body.get('hidden') is True)
                        data = {'id': key}
                    elif self.path == '/api/files/cancel':
                        with app.files_lock:
                            job = app.file_jobs.get(str(body.get('job', '')))
                            if job:
                                job['cancel'].set()
                        data = {'ok': True}
                    elif self.path == '/api/files/analyze':
                        data = {'count': app.select_local_documents(body)}
                    else:
                        raise ValueError('Pedido não encontrado')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as error:
                    print(f'[arquivos] {type(error).__name__}: {error}')
                    try:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': str(error)}, ensure_ascii=False).encode())
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                return
            if self.path == '/api/voice':
                if not secrets.compare_digest(self.headers.get('X-Dida-CSRF', ''), app.VOICE_CSRF):
                    self.send_error(403, 'Reabra as configurações')
                    self.close_connection = True
                    return
                try:
                    body = json.loads(raw_body.decode('utf-8'))
                    settings = app.validate_voice_settings(body)
                    available = {option['id'] for option in app.voice_catalog() if option['available']}
                    if settings['voice'] not in available:
                        raise ValueError('Esta voz não está instalada. Escolha outra opção.')
                    if body.get('preview') is not True:
                        settings = app.save_voice_settings(settings)
                    app.estado.interrupt_flag.set()
                    app.estado.fala.parar()
                    app.broadcast({'type': 'stopped'})
                    if body.get('preview') is True:
                        app.estado.fala.falar('Olá. Esta é uma amostra da minha voz. Podemos conversar no seu ritmo.', settings=settings)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(json.dumps({'ok': True, 'settings': settings}).encode())
                except (ValueError, TypeError, OSError) as e:
                    try:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': str(e)}, ensure_ascii=False).encode())
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == '/api/profile':
                try:
                    if length > 8192:
                        raise ValueError('Perfil muito longo')
                    body = json.loads(raw_body.decode('utf-8'))
                    if not isinstance(body, dict):
                        raise ValueError('Perfil invalido')
                    current = app.profile_snapshot()
                    app.save_profile(str(body.get('name', current['name'])), str(body.get('about', current['about'])), str(body.get('interests', current['interests'])), bool(body.get('intro_seen', current['intro_seen'])))
                    if 'interests' in body and str(body['interests']).strip():
                        threading.Thread(target=app.share_daily_curiosity, daemon=True).start()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(json.dumps({'ok': True}, ensure_ascii=False).encode())
                except Exception as e:
                    try:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                    except BrokenPipeError:
                        pass
                return
            if self.path == '/api/sens':
                body = raw_body.decode()
                try:
                    v = app.validate_sensitivity(json.loads(body).get('v', 0.5))
                    app.estado.sensibilidade = v
                    app.save_sens(v)
                    try:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(b'{"ok":true}')
                    except BrokenPipeError:
                        pass
                except Exception as e:
                    try:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                    except Exception:
                        pass
                return
            if self.path == '/api/msg':
                body = raw_body.decode()
                try:
                    j = json.loads(body)
                    txt = j.get('texto', '')
                    if not isinstance(txt, str) or not txt.strip():
                        raise ValueError('Mensagem vazia ou inválida')
                    txt = txt.strip()
                    route = app.classify_query(txt, False) if txt else 'fast'
                    if txt:
                        app.submit_message(txt)
                    try:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'ok': True, 'route': route}).encode())
                    except BrokenPipeError:
                        pass
                except Exception as e:
                    try:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(str(e).encode())
                    except Exception:
                        pass
                return
            if self.path == '/anexo':
                ctype = self.headers.get('Content-Type', '')
                try:
                    raw = raw_body
                    image_b64 = None
                    mime = None
                    parsed_txt = ''
                    texto = ''
                    fname = 'arquivo'
                    upload_received = False
                    if 'application/json' in ctype:
                        j = json.loads(raw.decode('utf-8'))
                        texto = str(j.get('texto', '') or '')
                        fname = str(j.get('fname', 'arquivo') or 'arquivo')
                        mime = str(j.get('mime', '') or '')
                        b64 = j.get('data', '') or ''
                        try:
                            data = base64.b64decode(b64, validate=True)
                            upload_received = True
                        except Exception as e:
                            raise ValueError(f'base64 invalido: {e}')
                        print(f'[anexo] JSON recebido: {fname} ({len(data)} bytes, mime={mime!r})')
                        if len(data) > app.MAX_FILE:
                            raise ValueError('Arquivo maior que 20 MB')
                        parsed_txt, image_b64, detected_mime = app.parse_file(fname, data)
                        if image_b64 is None and mime.startswith('image/'):
                            image_b64 = b64
                            detected_mime = mime or 'image/jpeg'
                        mime = detected_mime or mime
                    else:
                        from email import policy as _policy
                        from email.parser import BytesParser as _BytesParser
                        envelope = f'Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n'.encode() + raw
                        form = _BytesParser(policy=_policy.default).parsebytes(envelope)
                        if not form.is_multipart():
                            raise ValueError('Formato de anexo invalido')
                        fitem = None
                        for part in form.iter_parts():
                            field = part.get_param('name', header='content-disposition')
                            value = part.get_payload(decode=True) or b''
                            if field == 'texto':
                                texto = value.decode(part.get_content_charset() or 'utf-8', errors='replace')
                            elif field == 'file' and part.get_filename():
                                fitem = (part.get_filename(), value)
                        if fitem is not None:
                            fname, data = fitem
                            upload_received = True
                            print(f'[anexo] multipart recebido: {fname} ({len(data)} bytes)')
                            if len(data) > app.MAX_FILE:
                                raise ValueError('Arquivo maior que 20 MB')
                            parsed_txt, image_b64, mime = app.parse_file(fname, data)
                    route = app.classify_query(texto + ' ' + parsed_txt, image_b64 is not None)
                    if upload_received:
                        with app.attachment_lock:
                            safe_name = app.remember_attachment(fname, parsed_txt, len(data))
                            uploaded_attachment = app.last_attachment.copy()
                        app.broadcast({'type': 'sys', 'text': f'Arquivo recebido: {safe_name}. {len(parsed_txt)} caracteres disponíveis para análise nesta sessão.'})
                        full = f'[Anexo recebido: {safe_name}]\n' + (texto.strip() or 'Resuma o arquivo anexado e diga o que consegue analisar no trecho recebido.')
                    else:
                        full = texto.strip()
                    print(f"[anexo] route={route} image={('sim' if image_b64 else 'nao')} texto_len={len(texto)} parsed_len={len(parsed_txt)}")
                    if image_b64:
                        app.submit_message(full or 'Analise esta imagem.', image_b64, mime or 'image/jpeg', attachment=uploaded_attachment if upload_received else None)
                    elif full:
                        app.submit_message(full, attachment=uploaded_attachment if upload_received else None)
                    else:
                        print('[anexo] nada para enviar (sem texto e sem imagem)')
                    try:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'ok': True, 'route': route, 'image': bool(image_b64)}).encode())
                    except BrokenPipeError:
                        pass
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    try:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
                    except Exception:
                        pass
                return
            self.send_response(404)
            self.end_headers()
    return Handler
