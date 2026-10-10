"""Real HTTP and subprocess workflow, using a deliberately simulated LLM."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import requests
from dida_app.local_model import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]


class FakeModelHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'data':[{'id':MODEL_NAME,'architecture':{'input_modalities':['text']}}]}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path != '/v1/chat/completions' or body.get('model') != MODEL_NAME:
            self.send_error(400)
            return
        self.server.payloads.append(body)
        prompt = str(body['messages'][-1]['content'])
        first = 'primeira' in prompt.lower()
        before_headers = 'antes dos cabeçalhos' in prompt.lower()
        if before_headers:
            self.server.first_started.set()
            self.server.release_first.wait(5)
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.end_headers()
        try:
            for word in ['Parcial antiga. ' if first else 'Olá. ', 'Resposta de teste.']:
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':word}}]})+'\n\n').encode())
                self.wfile.flush()
                if first:
                    self.server.first_started.set()
                    self.server.release_first.wait(5)
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='dida_e2e_')
        cls.model = ThreadingHTTPServer(('127.0.0.1',0),FakeModelHandler)
        cls.model.daemon_threads = True
        cls.model.payloads = []
        cls.model.first_started = threading.Event()
        cls.model.release_first = threading.Event()
        cls.model_thread = threading.Thread(target=cls.model.serve_forever,daemon=True)
        cls.model_thread.start()
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1',0))
            port = reservation.getsockname()[1]
        cls.base = f'http://127.0.0.1:{port}'
        env = os.environ.copy()
        env.update(DIDA_CONFIG_DIR=cls.temp.name, DIDA_LLM_BASE_URL=f'http://127.0.0.1:{cls.model.server_port}/v1')
        cls.app = subprocess.Popen([sys.executable,str(ROOT/'Ia-Completa-Hibrida.py'),'--headless','--port',str(port)],
                                   env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            try:
                if requests.get(cls.base+'/api/mem',timeout=1).status_code==200:
                    return
            except requests.RequestException:
                pass
            if cls.app.poll() is not None:
                raise AssertionError('Application exited during startup')
            time.sleep(.05)
        cls.tearDownClass()
        raise AssertionError('Application startup timed out')

    @classmethod
    def tearDownClass(cls):
        cls.model.release_first.set()
        cls.app.send_signal(signal.SIGINT)
        try:
            cls.app.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.app.kill()
            cls.app.wait(timeout=3)
        cls.model.shutdown()
        cls.model.server_close()
        cls.model_thread.join()
        cls.temp.cleanup()

    def setUp(self):
        self.csrf=requests.get(self.base+'/api/mem',timeout=3).json()['csrf']
        self.post('/api/mem/clear',{})
        self.post('/api/mem/settings',{'enabled':True})

    def post(self,path,body):
        response=requests.post(self.base+path,json=body,headers={'X-Dida-CSRF':self.csrf},timeout=3)
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def read_answer(self,stream):
        for line in stream.iter_lines(chunk_size=1):
            if line.startswith(b'data: '):
                event=json.loads(line[6:])
                if event['type']=='bot':
                    return event['text']
        self.fail('No final response event')

    def test_text_response_saved_and_disable_clear_work(self):
        with requests.get(self.base+'/api/events',stream=True,timeout=(3,10)) as events:
            self.post('/api/msg',{'texto':'oi'})
            self.assertEqual(self.read_answer(events),'Olá. Resposta de teste.')
        history=requests.get(self.base+'/api/mem/history',timeout=3).json()
        self.assertEqual(history['count'],1)
        self.assertEqual(history['conversations'][0]['bot'],'Olá. Resposta de teste.')
        self.assertEqual(self.model.payloads[-1]['model'],MODEL_NAME)
        self.post('/api/mem/settings',{'enabled':False})
        with requests.get(self.base+'/api/events',stream=True,timeout=(3,10)) as events:
            self.post('/api/msg',{'texto':'mensagem privada'})
            self.assertEqual(self.read_answer(events),'Olá. Resposta de teste.')
        self.assertEqual(requests.get(self.base+'/api/mem',timeout=3).json()['count'],1)
        self.assertEqual(self.post('/api/mem/clear',{})['count'],0)

    def test_new_message_cancels_old_response(self):
        self.model.first_started.clear()
        self.model.release_first.clear()
        with requests.get(self.base+'/api/events',stream=True,timeout=(3,10)) as events:
            self.post('/api/msg',{'texto':'primeira'})
            self.assertTrue(self.model.first_started.wait(3))
            self.post('/api/msg',{'texto':'segunda'})
            self.model.release_first.set()
            self.assertEqual(self.read_answer(events),'Olá. Resposta de teste.')
        history=requests.get(self.base+'/api/mem/history',timeout=3).json()
        self.assertEqual(history['count'],1)
        self.assertEqual(history['conversations'][0]['user'],'segunda')

    def test_cancel_while_waiting_for_response_headers(self):
        self.model.first_started.clear()
        self.model.release_first.clear()
        try:
            with requests.get(self.base+'/api/events',stream=True,timeout=(3,4)) as events:
                self.post('/api/msg',{'texto':'antes dos cabeçalhos'})
                self.assertTrue(self.model.first_started.wait(2))
                start=time.monotonic()
                self.post('/api/msg',{'texto':'nova pergunta'})
                self.assertEqual(self.read_answer(events),'Olá. Resposta de teste.')
                self.assertLess(time.monotonic()-start,2)
        finally:
            self.model.release_first.set()

    def test_stop_and_memory_routes_require_csrf(self):
        for path in ['/api/stop','/api/mem/settings','/api/mem/clear']:
            response=requests.post(self.base+path,json={'enabled':False},timeout=3)
            self.assertEqual(response.status_code,403)

    def test_model_status_and_image_rejection(self):
        status=requests.get(self.base+'/api/models',timeout=3).json()
        self.assertTrue(status['available'])
        self.assertFalse(status['vision'])
        image=requests.post(self.base+'/anexo',json={'fname':'image.png','data':'aW1hZ2U='},timeout=3)
        self.assertEqual(image.status_code,400)
        self.assertIn('suporte a imagens',image.json()['error'])
