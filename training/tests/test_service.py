from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from service import ProgressTail
from serve_selfplay import create_handler


class ServiceTest(unittest.TestCase):
    def test_progress_tail_retries_incomplete_events_and_resets_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl'
            path.write_text(json.dumps(dict(event='learning_update',iteration=40,optimizer_steps_total=3729))+'\n'+ '{"event":"config",')
            tail=ProgressTail(path)
            self.assertEqual(tail.poll()['optimizer_steps'],3729)
            with path.open('a') as f:f.write('"config":{"optimizer_steps_start":3700}}\n')
            self.assertEqual(tail.poll()['optimizer_steps'],3700)
            with path.open('a') as f:f.write(json.dumps(dict(event='train_epoch',optimizer_steps=1))+'\n')
            self.assertEqual(tail.poll()['optimizer_steps'],3701)
            with path.open('a') as f:f.write(json.dumps(dict(event='learning_update',iteration=41,optimizer_steps_total=3701))+'\n')
            self.assertEqual(tail.poll()['optimizer_steps'],3701)

    def test_readonly_feed_serves_status_and_only_complete_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);web=root/'web';web.mkdir();(web/'index.html').write_text('viewer')
            feed=root/'feed';feed.mkdir();status=root/'status.json';status.write_text('{"state":"training","selfplay_games":1}')
            (feed/'game-0.meta.json').write_text('{"game":0}')
            (feed/'game-127.meta.json').write_text('{"game":127}')
            (feed/'game-0.jsonl').write_text('{"seed":12,"sequence":0}\n{"seed":12,"sequence":1}\n{"seed":')
            server=ThreadingHTTPServer(('127.0.0.1',0),create_handler(web,feed,status))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def get(path):
                with opener.open(f'http://127.0.0.1:{server.server_port}{path}', timeout=5) as response:return json.load(response)
            try:
                listing=get('/__selfplay/games')
                self.assertEqual(listing['service']['state'],'training')
                self.assertEqual([game['game'] for game in listing['games']],[0])
                result=get('/__selfplay/game/0?seed=12&after=0')
                self.assertFalse(result['reset']);self.assertEqual(len(result['rows']),1)
                self.assertEqual(result['rows'][0]['sequence'],1)
                result=get('/__selfplay/game/0?seed=11&after=5')
                self.assertTrue(result['reset']);self.assertEqual(len(result['rows']),2)
                with self.assertRaises(urllib.error.HTTPError):get('/__selfplay/game/../../status.json')
            finally:
                server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
