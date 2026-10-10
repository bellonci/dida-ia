import os, time, math, threading, queue, subprocess, shutil, tempfile
from pathlib import Path

class Fala:

    def __init__(self, app):
        self.app = app
        self.fila = queue.Queue()
        self.parada = threading.Event()
        self.ocupada = threading.Event()
        self._busy_lock = threading.RLock()
        self._generation = 0
        self._active = None
        self.level = 0.0
        self.disponivel = False
        self._shutdown = threading.Event()
        self._worker = None

    def initialize_audio(self):
        try:
            import pygame
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
            self.disponivel = True
        except Exception as e:
            print(f'[Fala] mixer {e}')

    def falar(self, texto: str, settings: dict=None):
        texto = self.app.speech_text(texto)
        if texto.strip() and self.disponivel:
            with self._busy_lock:
                self.ocupada.set()
                self.fila.put((self._generation, texto, (settings or self.app.voice_snapshot()).copy()))

    def _clear_if_idle(self):
        with self._busy_lock:
            if self.fila.empty() and self._active != self._generation:
                self.ocupada.clear()
                if hasattr(self.app, 'estado'):
                    self.app.estado.tts_tail_until = time.time() + 1.2

    def parar(self):
        with self._busy_lock:
            self._generation += 1
            self.parada.set()
            while True:
                try:
                    self.fila.get_nowait()
                    self.fila.task_done()
                except queue.Empty:
                    break
            try:
                if self.disponivel:
                    import pygame
                    pygame.mixer.stop()
                    pygame.mixer.music.stop()
            except Exception as e:
                print(f'[TTS stop] {e}')
            self.level = 0.0
            self._clear_if_idle()

    def _notify(self, event):
        coordinator = getattr(self.app, "requests_manager", None)
        if coordinator is None:
            with self._busy_lock:
                current = self._active == self._generation
            if current:
                self.app.broadcast(event)
            return
        # Always acquire the coordinator before the audio lock, as submit/cancel do.
        with coordinator.lock:
            with self._busy_lock:
                if self._active == self._generation:
                    self.app.broadcast(event)

    def _gerar_audio(self, texto: str, tmp: str, settings: dict) -> bool:
        selected = self.app.VOICE_OPTIONS[settings['voice']]
        speed = settings['speed']
        order = [selected['engine']]
        if selected['engine'] != 'local':
            order += [engine for engine in ('edge', 'google', 'local') if engine not in order]
        for index, engine in enumerate(order):
            if engine == 'google' and (not self.app.GTTS_OK):
                continue
            if engine == 'edge' and (not self.app.EDGE_OK):
                continue
            if index:
                self._notify({'type': 'status', 'text': 'Voz escolhida indisponível; tentando uma alternativa.'})
            try:
                Path(tmp).write_bytes(b'')
                Path(tmp).with_suffix('.wav').unlink(missing_ok=True)
                if engine == 'google':
                    self.app.gTTS(text=texto, lang=self.app.VOICE_LANG, tld=self.app.VOICE_TLD, timeout=10).save(tmp)
                    if abs(speed - 1.0) > 0.001:
                        ffmpeg = shutil.which('ffmpeg')
                        if ffmpeg:
                            subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', tmp, '-af', f'atempo={speed}', str(Path(tmp).with_suffix('.wav'))], check=True, capture_output=True, timeout=15)
                        else:
                            self._notify({'type': 'sys', 'text': 'Google usa velocidade normal: o ajuste precisa do FFmpeg.'})
                elif engine == 'edge':
                    import asyncio
                    rate = f'{round((speed - 1) * 100):+d}%'
                    voice = selected.get('voice', 'pt-BR-FranciscaNeural')

                    async def generate():
                        await self.app.edge_tts.Communicate(texto, voice=voice, rate=rate).save(tmp)

                    async def bounded():
                        await asyncio.wait_for(generate(), timeout=15)
                    asyncio.run(bounded())
                else:
                    import pyttsx3
                    eng = pyttsx3.init()
                    eng.setProperty('rate', round(160 * speed))
                    eng.save_to_file(texto, str(Path(tmp).with_suffix('.wav')))
                    eng.runAndWait()
                    eng.stop()
                target = Path(tmp).with_suffix('.wav')
                if not target.exists():
                    target = Path(tmp)
                if target.exists() and target.stat().st_size > 0:
                    return True
            except Exception as e:
                print(f'[TTS {engine}] {type(e).__name__}: {e}')
        return False

    def close(self):
        self._shutdown.set()
        self.parar()
        if self._worker is not None:
            self._worker.join(timeout=1)

    def start(self):
        if self._worker is not None and self._worker.is_alive():
            return
        if not self.disponivel:
            print('[TTS] audio desativado; respostas continuam disponiveis na interface')
            return
        if not (self.app.GTTS_OK or self.app.EDGE_OK):
            print('[TTS] nenhum motor de voz instalado. Instale: pip install gTTS edge-tts')

        def loop():
            try:
                import pygame
            except Exception as e:
                print(f'[TTS] pygame indisponivel: {e}')
                return
            while not self._shutdown.is_set():
                try:
                    generation, texto, settings = self.fila.get(timeout=0.2)
                except queue.Empty:
                    continue
                tmp = None
                try:
                    with self._busy_lock:
                        if generation != self._generation:
                            continue
                        self._active = generation
                        self.parada.clear()
                    fd, tmp = tempfile.mkstemp(prefix='dida_tts_', suffix='.mp3')
                    os.close(fd)
                    if not self._gerar_audio(texto, tmp, settings):
                        raise RuntimeError('Nenhum motor conseguiu gerar áudio')
                    audio_path = str(Path(tmp).with_suffix('.wav')) if Path(tmp).with_suffix('.wav').exists() else tmp
                    with self._busy_lock:
                        if generation != self._generation:
                            continue
                        pygame.mixer.music.load(audio_path)
                        pygame.mixer.music.set_volume(settings['volume'])
                        pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        with self._busy_lock:
                            if generation != self._generation:
                                pygame.mixer.music.stop()
                                break
                        self.level = 0.5 + 0.5 * abs(math.sin(time.time() * 9))
                        time.sleep(0.05)
                    pause_until = time.monotonic() + settings['pause']
                    while time.monotonic() < pause_until:
                        with self._busy_lock:
                            if generation != self._generation:
                                break
                        time.sleep(0.02)
                except Exception as e:
                    print(f'[TTS] {type(e).__name__}: {e}')
                    if hasattr(self.app, 'estado'):
                        self._notify({'type': 'sys', 'text': 'Não consegui reproduzir a voz. A resposta continua na tela.'})
                finally:
                    self.level = 0.0
                    try:
                        pygame.mixer.music.unload()
                    except Exception:
                        pass
                    if tmp:
                        for file in (Path(tmp), Path(tmp).with_suffix('.wav')):
                            try:
                                file.unlink(missing_ok=True)
                            except OSError as e:
                                print(f'[TTS cleanup] {e}')
                    with self._busy_lock:
                        self._active = None
                        self._clear_if_idle()
                    self.fila.task_done()
        self._worker = threading.Thread(target=loop, daemon=True)
        self._worker.start()
