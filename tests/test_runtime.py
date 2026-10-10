from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
from dida_app.audio import Fala
from dida_app.context import attachment_excerpt, document_excerpt, limit_messages, message_tokens, text_tokens
from dida_app.events import EventHub
from dida_app.local_model import LocalModelConfig, MODEL_NAME
from dida_app.memory import Memory
from dida_app.models import ModelCatalog
from dida_app.requests import CancellationToken, RequestCoordinator
from test_dida import dida


class CancellationTests(unittest.TestCase):
    def test_dequeued_old_request_cannot_activate(self):
        manager = RequestCoordinator()
        manager.submit('old')
        old = manager.queue.get_nowait()
        manager.submit('new')
        self.assertFalse(manager.activate(old))
        manager.queue.task_done()
        new = manager.queue.get_nowait()
        self.assertTrue(manager.activate(new))
        manager.queue.task_done()
        self.assertEqual(manager.queue.unfinished_tasks, 0)

    def test_clear_global_event_does_not_revive_old_token(self):
        manager = RequestCoordinator()
        old = manager.submit('old')
        manager.submit('new')
        action = Mock()
        self.assertFalse(manager.run_if_current(old, action))
        action.assert_not_called()

    def test_cancel_all_discards_pending(self):
        manager = RequestCoordinator()
        token = manager.submit('old')
        manager.cancel_all()
        self.assertTrue(token.cancelled())
        self.assertTrue(manager.queue.empty())
        self.assertEqual(manager.queue.unfinished_tasks, 0)

    def test_socket_close_does_not_block_submitter(self):
        token = CancellationToken()
        release = threading.Event()
        entered = threading.Event()
        resource = Mock()
        resource.close.side_effect = lambda: (entered.set(), release.wait(2))
        token.register(resource)
        start = time.monotonic()
        token.cancel()
        try:
            self.assertLess(time.monotonic() - start, 0.5)
            self.assertTrue(entered.wait(1))
            self.assertTrue(token.cancelled())
        finally:
            release.set()

    def test_cancelled_response_never_publishes_or_saves(self):
        manager = RequestCoordinator()
        token = manager.submit('old')
        request = manager.queue.get_nowait()
        manager.activate(request)
        memory = Mock()
        events = EventHub()
        client = events.subscribe()
        def stream(*args, **kwargs):
            yield 'Partial answer. '
            manager.submit('new')
            yield 'stale answer'
        dida.response_context.token = token
        with patch.object(dida, 'requests_manager', manager), patch.object(dida, 'memory', memory), patch.object(dida.estado, 'events', events), patch.object(dida, 'search_context', return_value=''), patch.object(dida, 'call_llm_stream', side_effect=stream):
            try:
                dida.process_msg('old')
            finally:
                dida.response_context.token = None
                manager.queue.task_done()
        memory.add.assert_not_called()
        delivered = []
        while not client.queue.empty():
            delivered.append(client.receive())
        self.assertFalse(any(event['type'] == 'bot' for event in delivered))
        self.assertFalse(any('stale' in event.get('text','') for event in delivered))


class EventTests(unittest.TestCase):
    def test_slow_subscriber_does_not_block_other_clients(self):
        hub = EventHub(capacity=2)
        slow = hub.subscribe()
        fast = hub.subscribe()
        for i in range(3):
            hub.publish({'type': 'bot_delta', 'text': str(i)})
            self.assertEqual(fast.receive()['text'], str(i))
        self.assertTrue(slow.closed.is_set())
        self.assertFalse(fast.closed.is_set())

    def test_client_limit_and_cleanup(self):
        hub = EventHub(max_clients=1)
        client = hub.subscribe()
        with self.assertRaises(ValueError):
            hub.subscribe()
        hub.unsubscribe(client)
        new = hub.subscribe()
        hub.close()
        self.assertTrue(new.closed.is_set())


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'memory.json'
        self.memory = Memory(self.path)

    def test_disable_prevents_saving_and_context_retrieval(self):
        self.memory.add('old message', 'old answer')
        self.memory.set_enabled(False)
        self.memory.add('private message', 'private answer')
        self.assertEqual(self.memory.status(), {'enabled': False, 'count': 1})
        self.assertEqual(self.memory.recent_messages(), [])
        self.assertEqual(self.memory.search('old message'), '')
        self.assertNotIn('private message', self.path.read_text())
        restored = Memory(self.path)
        self.assertFalse(restored.enabled)
        restored.set_enabled(True)
        self.assertEqual(len(restored.history()), 1)

    def test_clear_deletes_history_and_invalid_backups(self):
        self.memory.add('old', 'answer')
        backup = self.path.with_suffix('.invalid-123.json')
        backup.write_text('sensitive old content')
        self.memory.clear()
        self.assertFalse(backup.exists())
        self.assertEqual(Memory(self.path).history(), [])

    def test_errors_and_cancellations_not_saved(self):
        for bot, meta in [('Não consegui gerar uma resposta agora. Verifique o modelo.', {}), ('Partial response', {'cancelled': True}), ('Error', {'error': True})]:
            self.memory.add('message', bot, meta)
        self.assertEqual(self.memory.status()['count'], 0)

    def test_history_is_not_mutable_outside_lock(self):
        self.memory.add('user', 'answer')
        history = self.memory.history()
        history[0]['user'] = 'changed'
        self.assertEqual(self.memory.history()[0]['user'], 'user')


class ContextTests(unittest.TestCase):
    def test_relevant_document_section_beyond_old_character_cutoff(self):
        text = 'Introdução genérica.\n' * 2000 + '\nRESULTADO: receita trimestral aumentou 42%.\n'
        excerpt = document_excerpt(text, 'Qual foi a receita trimestral?', budget=1800)
        self.assertIn('42%', excerpt)
        self.assertLessEqual(text_tokens(excerpt), 1800)
        self.assertIn('não enviado na íntegra', excerpt)

    def test_every_selected_file_has_context(self):
        attachment = {'documents': [{'filename': f'{i}.txt', 'content_excerpt': 'conteúdo '*3000} for i in range(10)]}
        excerpt = attachment_excerpt(attachment, 'resuma tudo', budget=3500)
        for i in range(10):
            self.assertIn(f'{i}.txt', excerpt)
        self.assertLessEqual(text_tokens(excerpt), 3500)

    def test_unicode_and_history_stay_within_budget(self):
        messages = [{'role': 'system', 'content': 'Instruções '*500},
                    {'role': 'user', 'content': 'old '*2000},
                    {'role': 'assistant', 'content': 'old answer '*1000},
                    {'role': 'user', 'content': 'Pergunta atual: ação, café e coração. '*1000}]
        result = limit_messages(messages, budget=1400)
        self.assertEqual([row['role'] for row in result], ['system', 'user'])
        self.assertTrue(result[-1]['content'].startswith('Pergunta atual'))
        self.assertLessEqual(sum(message_tokens(row) for row in result), 1400)
        self.assertNotIn('\ufffd', result[-1]['content'])
        self.assertEqual(len(messages), 4)

    def test_image_preserved_and_reserved(self):
        messages = [{'role': 'system', 'content': 'instructions'}, {'role': 'user', 'content': [
            {'type': 'text', 'text': 'question '*1000}, {'type': 'image_url','image_url':{'url':'data:image/png;base64,exact'}}]}]
        result = limit_messages(messages, budget=5000)
        self.assertEqual(result[-1]['content'][1], messages[-1]['content'][1])
        self.assertLessEqual(sum(message_tokens(row) for row in result), 5000)
        with self.assertRaises(ValueError):
            limit_messages(messages, budget=2048)


class LocalModelTests(unittest.TestCase):
    def config(self, path=None):
        return LocalModelConfig(path or Path('/tmp') / MODEL_NAME)

    def test_remote_endpoint_and_wrong_file_rejected(self):
        for endpoint in ['https://api.groq.com/v1', 'http://evil.example/v1', 'http://user:pass@localhost/v1']:
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                LocalModelConfig(Path('/tmp') / MODEL_NAME, endpoint)
        with self.assertRaises(ValueError):
            LocalModelConfig(Path('/tmp/wrong.gguf'))

    def test_catalog_requires_exact_model(self):
        response = Mock()
        response.json.return_value = {'data': [{'id':'another-model'}]}
        catalog = ModelCatalog(self.config(), get=Mock(return_value=response))
        self.assertFalse(catalog.refresh())
        self.assertFalse(catalog.supports_vision())
        response.close.assert_called()

    def test_catalog_caches_and_does_not_infer_vision(self):
        response = Mock()
        response.json.return_value = {'data': [{'id':MODEL_NAME}]}
        get = Mock(return_value=response)
        catalog = ModelCatalog(self.config(), get=get)
        self.assertTrue(catalog.refresh())
        self.assertFalse(catalog.supports_vision())
        self.assertEqual(catalog.status()['model'], MODEL_NAME)
        get.assert_called_once()

    def test_catalog_failure_clears_stale_availability(self):
        response = Mock()
        response.json.return_value = {'data': [{'id':MODEL_NAME}]}
        get = Mock(side_effect=[response, requests.ConnectionError('offline')])
        catalog = ModelCatalog(self.config(), get=get)
        self.assertTrue(catalog.refresh())
        self.assertFalse(catalog.refresh(force=True))
        self.assertFalse(catalog.status()['available'])

    def test_command_has_exact_model_alias_and_local_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / MODEL_NAME
            path.write_bytes(b'GGUFtest-header')
            command = self.config(path).command('/usr/bin/true')
            self.assertEqual(command[command.index('--alias')+1], MODEL_NAME)
            self.assertEqual(command[command.index('--host')+1], '127.0.0.1')
            path.write_text('not a GGUF')
            with self.assertRaises(ValueError):
                self.config(path).command('/usr/bin/true')

    def test_missing_model_has_actionable_failure(self):
        with self.assertRaises(FileNotFoundError):
            self.config(Path('/tmp/missing-model') / MODEL_NAME).command('/usr/bin/true')

    def test_incomplete_stream_not_saved(self):
        response = Mock()
        response.iter_lines.return_value = [b'data: {"choices":[{"delta":{"content":"Partial"}}]}']
        dida.response_context.failed = False
        dida.estado.interrupt_flag.clear()
        with patch.object(dida.model_catalog,'refresh',return_value=True), patch('dida_app.llm.open_stream', return_value=response):
            self.assertEqual(list(dida.local_chat_stream([])), ['Partial'])
        self.assertTrue(dida.response_context.failed)
        response.close.assert_called_once()


class VoiceTests(unittest.TestCase):
    def test_cancelled_synthesis_cannot_resume_old_playback(self):
        runtime = SimpleNamespace(speech_text=lambda text:text, voice_snapshot=lambda:{'voice':'local','speed':1,'volume':1,'pause':0},
                                  GTTS_OK=False,EDGE_OK=False,broadcast=Mock())
        voice = Fala(runtime)
        voice.disponivel = True
        pygame = Mock()
        pygame.mixer.music.get_busy.return_value = False
        entered = threading.Event()
        release = threading.Event()
        played = threading.Event()
        pygame.mixer.music.play.side_effect = played.set
        def synthesize(text, path, settings):
            if text == 'old':
                entered.set()
                release.wait(2)
            Path(path).write_bytes(b'audio')
            return True
        voice._gerar_audio = synthesize
        with patch.dict('sys.modules', {'pygame':pygame}):
            try:
                voice.start()
                voice.falar('old')
                self.assertTrue(entered.wait(1))
                voice.parar()
                voice.falar('new')
                release.set()
                self.assertTrue(played.wait(2))
                self.assertEqual(pygame.mixer.music.play.call_count,1)
            finally:
                release.set()
                voice.close()

    def test_local_transcription_does_not_download_a_model(self):
        with patch.dict('os.environ', {'DIDA_STT_MODEL_PATH':''}), patch('requests.post') as post:
            self.assertIsNone(dida.transcribe_audio(dida.np.zeros(100)))
            post.assert_not_called()
