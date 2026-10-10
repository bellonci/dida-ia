import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('model_downloader', ROOT/'scripts/download_model.py')
downloader=importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def get(self, data, sha=None):
        digest=sha or hashlib.sha256(data).hexdigest()
        metadata={'sha':'a'*40,'siblings':[{'rfilename':downloader.MODEL_NAME,'lfs':{'sha256':digest,'size':len(data)}}]}
        def response(value=None):
            result=Mock()
            result.__enter__=Mock(return_value=result)
            result.__exit__=Mock(return_value=False)
            result.json.return_value=value
            return result
        first=response(metadata)
        second=response()
        second.iter_content.return_value=[data]
        return Mock(side_effect=[first,second])

    def test_verified_download_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/downloader.MODEL_NAME
            data=b'GGUFtest-model'
            get=self.get(data)
            self.assertEqual(downloader.download(path,get),path)
            self.assertEqual(path.read_bytes(),data)
            self.assertIn('/resolve/'+'a'*40+'/',get.call_args.args[0])
            again=self.get(data)
            downloader.download(path,again)
            self.assertEqual(again.call_count,1)
            self.assertEqual(list(path.parent.glob('*.partial')),[])

    def test_bad_checksum_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/downloader.MODEL_NAME
            with self.assertRaises(ValueError):
                downloader.download(path,self.get(b'GGUFdata',sha='0'*64))
            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob('*.partial')),[])

    def test_existing_user_file_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/downloader.MODEL_NAME
            path.write_bytes(b'user file')
            with self.assertRaises(ValueError):
                downloader.download(path,self.get(b'GGUFdata'))
            self.assertEqual(path.read_bytes(),b'user file')

    def test_parallel_ranges_preserve_existing_prefix(self):
        data=b'GGUF0123456789abcdefghijklmnop'
        def get(url,headers,**kwargs):
            start,end=map(int,headers['Range'][6:].split('-'))
            response=Mock(status_code=206)
            response.__enter__=Mock(return_value=response)
            response.__exit__=Mock(return_value=False)
            response.headers={'Content-Range':f'bytes {start}-{end}/{len(data)}'}
            response.iter_content.return_value=[data[start:end+1]]
            return response
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'download.partial'
            path.touch()
            prefix=Path(directory)/'old.partial'
            prefix.write_bytes(data[:8])
            with prefix.open('rb') as source:
                downloader.download_ranges('https://example.com/model',path,len(data),get=get,chunk_size=5,prefix=(source,8))
            self.assertEqual(path.read_bytes(),data)

    def test_unconfirmed_range_rejected(self):
        response=Mock(status_code=200)
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        response.headers={}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'download.partial'
            path.touch()
            with self.assertRaises(ValueError):
                downloader.download_ranges('https://example.com/model',path,100,get=Mock(return_value=response))
