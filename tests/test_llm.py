"""Offline tests for llm.py with a fake Colab CLI, fake ssh and a mock server (Linux, about 80 s). run: python3 -B -m unittest -v tests/test_llm.py"""
import http.client, importlib, json, os, shutil, signal, subprocess, sys, tarfile, tempfile, threading, time, unittest

sys.dont_write_bytecode = True
TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
LLM = os.path.join(ROOT, 'llm.py')
VM_DIR = os.path.join(ROOT, 'vm')
PY = sys.executable
FAKE_COLAB = '%s -B %s' % (PY, os.path.join(TESTS, 'fake_colab.py'))
FAKE_SSH = '%s -B %s' % (PY, os.path.join(TESTS, 'fake_ssh.py'))
HF_SECRET = 'hf_FAKE0123456789'
TMP, BASE_STATE, COUNTER = None, None, [0]


def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLM_', 'FAKE_')) and k not in ('HF_TOKEN', 'CTX', 'MMPROJ')}
    env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep) if '.local/bin' not in p)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def kill_matching(token):
    """Kills every process of this user whose environment mentions token (the per-run temp dir)."""
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open('/proc/%s/environ' % d, 'rb') as f:
                if token.encode() in f.read():
                    os.kill(int(d), signal.SIGKILL)
        except OSError:
            pass


def procs_with(token, word):
    out = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open('/proc/%s/environ' % d, 'rb') as f:
                env = f.read()
            with open('/proc/%s/cmdline' % d, 'rb') as f:
                cmd = f.read()
            with open('/proc/%s/stat' % d) as f:
                zombie = f.read().rsplit(')', 1)[1].split()[0] == 'Z'
        except OSError:
            continue
        if token.encode() in env and word.encode() in cmd and not zombie:
            out.append(int(d))
    return out


def read(path):
    with open(path, errors='replace') as f:
        return f.read()


def setUpModule():
    global TMP, BASE_STATE
    root = os.environ.get('LLM_TEST_TMP')
    if root:
        os.makedirs(root, exist_ok=True)
    TMP = tempfile.mkdtemp(prefix='run-', dir=root)
    BASE_STATE = os.path.join(TMP, 'state0')
    os.makedirs(os.path.join(TMP, 'fake0'))
    env = dict(clean_env(), LLM_STATE=BASE_STATE, HOME=os.path.join(TMP, 'home'), LLM_COLAB=FAKE_COLAB, FAKE_COLAB_DIR=os.path.join(TMP, 'fake0'))
    r = subprocess.run([PY, '-B', LLM, 'setup'], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and 'Setup done' in r.stdout, r.stdout + r.stderr


def tearDownModule():
    kill_matching(TMP)
    shutil.rmtree(TMP, ignore_errors=True)


class Resp:
    def __init__(self, status, t0, t_headers, chunks, headers):
        self.status, self.t0, self.t_headers, self.chunks, self.headers = status, t0, t_headers, chunks, headers
        self.raw = b''.join(c for t, c in chunks)
        self.lines = self.raw.decode('utf-8', 'replace').split('\n')
        self.comments = [l for l in self.lines if l.startswith(':')]
        self.data = [l[6:] for l in self.lines if l.startswith('data: ')]

    def content(self):
        if not self.data:
            return json.loads(self.raw)['choices'][0]['message']['content']
        out = ''
        for d in self.data:
            if d != '[DONE]':
                for ch in json.loads(d).get('choices', []):
                    out += ch['delta'].get('content') or ''
        return out


class Case:
    def __init__(self, test, **env):
        COUNTER[0] += 1
        n = COUNTER[0]
        self.dir = tempfile.mkdtemp(prefix='%s-' % test.id().split('.')[-1], dir=TMP)
        self.state, self.fake = os.path.join(self.dir, 'state'), os.path.join(self.dir, 'fake')
        shutil.copytree(BASE_STATE, self.state)
        os.makedirs(self.fake)
        self.port, self.vm_port, self.name = 28100 + 3 * n, 28101 + 3 * n, 'case%d' % n
        assert self.port + 2 < 28500
        self.env = dict(clean_env(), LLM_STATE=self.state, LLM_COLAB=FAKE_COLAB, LLM_SSH_CMD=FAKE_SSH, LLM_SESSION=self.name,
                        LLM_PORT=str(self.port), LLM_TUNNEL_PORT=str(28500 + 5 * n), LLM_POLL_S='0.3', LLM_IDLE_MIN='10',
                        LLM_MAX_HOURS='1', LLM_VM_DIR=VM_DIR, FAKE_COLAB_DIR=self.fake, FAKE_VM_PORT=str(self.vm_port),
                        FAKE_VM_INNER_PORT=str(self.vm_port + 1),
                        FAKE_BOOT_S='1.5', HOME=os.path.join(self.dir, 'home'))
        self.env.update(env)
        assert 'fake_colab.py' in self.env['LLM_COLAB'] and 'fake_ssh.py' in self.env['LLM_SSH_CMD']
        assert all(28000 <= int(self.env[k]) <= 28999 for k in ('LLM_PORT', 'LLM_TUNNEL_PORT', 'FAKE_VM_PORT', 'FAKE_VM_INNER_PORT'))
        self.key = read(os.path.join(self.state, 'api_key')).strip()
        self.procs, self.outs = [], []
        test.addCleanup(self.cleanup)

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                pass
        kill_matching(self.dir)
        for f in self.outs:
            f.close()

    def llm(self, *args, timeout=60, input=None):
        return subprocess.run([PY, '-B', LLM] + list(args), env=self.env, input=input, capture_output=True, text=True, timeout=timeout)

    def serve(self):
        out = open(os.path.join(self.dir, 'serve%d.out' % len(self.procs)), 'w')
        self.outs.append(out)
        p = subprocess.Popen([PY, '-B', LLM, 'serve'], env=self.env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append(p)
        self.wait(lambda: self.health_status() is not None or p.poll() is not None, 10)
        assert p.poll() is None, read(out.name)
        return p

    def health_status(self):
        try:
            return self.req(method='GET', path='/health', timeout=3).status
        except OSError:
            return None

    def req(self, stream=False, path='/v1/chat/completions', auth=True, method='POST', body=None, timeout=60, host=None, extra_headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        headers = {'Content-Type': 'application/json'}
        if host:
            headers['Host'] = host
        if auth:
            headers['Authorization'] = 'Bearer ' + self.key
        if extra_headers:
            headers.update(extra_headers)
        if method == 'POST' and body is None:
            body = {'model': 'DeepSeek-V4.1-Flash-1017-UD-IQ1_M', 'messages': [{'role': 'user', 'content': 'Hello World!'}], 'stream': stream}
        t0 = time.time()
        conn.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=headers)
        r = conn.getresponse()
        th, chunks = time.time() - t0, []
        while True:
            d = r.read1(65536)
            if not d:
                break
            chunks.append((time.time() - t0, d))
        conn.close()
        return Resp(r.status, t0, th, chunks, dict(r.getheaders()))

    def wait(self, fn, timeout, step=0.1):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if fn():
                return time.time() - t0
            time.sleep(step)
        raise AssertionError('condition not met in %.0f s' % timeout)

    def lines(self, f):
        try:
            with open(os.path.join(self.fake, f)) as fh:
                return [json.loads(l) for l in fh if l.strip()]
        except FileNotFoundError:
            return []

    def calls(self, cmd=None):
        return [c['argv'] for c in self.lines('calls.log') if cmd is None or c['argv'][0] == cmd]

    def sessions(self):
        try:
            with open(os.path.join(self.fake, 'sessions.json')) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def session_json(self):
        try:
            with open(os.path.join(self.state, 'session.json')) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def res(self, f):
        return os.path.join(self.fake, 'vms', self.name, 'content', 'res', f)

    def log(self):
        text = ''
        for f in [os.path.join(self.state, 'llm.log')] + [os.path.join(self.dir, x) for x in os.listdir(self.dir) if x.endswith('.out')]:
            if os.path.exists(f):
                text += read(f)
        return text

    def gate_at(self):
        return float(read(self.res('gate_open_at')))

    def assert_no_forward_before_gate(self, test):
        fwd, gate = self.lines('fwd.log'), self.gate_at()
        test.assertTrue([f for f in fwd if f['ok']])
        test.assertEqual([f for f in fwd if f['ok'] and f['t'] < gate], [])
        return [f for f in fwd if not f['ok'] and f['t'] < gate]

    def assert_all_stopped(self, test):
        test.assertEqual(self.sessions(), {})
        self.wait(lambda: not procs_with(self.dir, '_watchdog') and not procs_with(self.dir, 'fake_ssh.py'), 10)
        test.assertEqual(self.lines('violations.log'), [])


MODEL_ID = 'DeepSeek-V4.1-Flash-1017-UD-IQ1_M'


class ProxyTests(unittest.TestCase):
    def test_01_stream_first_request(self):
        c = Case(self, FAKE_BOOT_S='6')
        c.serve()
        self.assertEqual(c.health_status(), 503)
        self.assertEqual(c.req(stream=True, auth=False).status, 401)
        self.assertEqual(c.calls('new'), [])
        r = c.req(stream=True)
        self.assertEqual(r.status, 200)
        self.assertLess(r.t_headers, 1.0)
        self.assertTrue(r.chunks[0][1].startswith(b': starting gpu vm (0 s) - '), r.chunks[0])
        self.assertLess(r.chunks[0][0], 1.0)
        self.assertTrue(any(cm.startswith(': starting gpu vm (5 s) - ') for cm in r.comments), r.comments)
        self.assertLess(r.raw.index(b': starting gpu vm (5 s)'), r.raw.index(b'data: '))
        self.assertEqual(r.content(), 'ok')
        self.assertEqual(r.data[-1], '[DONE]')
        self.assertLess(c.lines('ssh_calls.log')[0]['t'], float(read(c.res('first_token_at'))))
        self.assertTrue(c.assert_no_forward_before_gate(self))
        order = [a[0] for a in c.calls() if a[0] in ('usage', 'new', 'upload')]
        self.assertEqual(order[:3], ['usage', 'new', 'upload'])
        self.assertEqual(len(c.calls('new')), 1)
        self.assertEqual(len(procs_with(c.dir, '_watchdog')), 1)
        check = json.loads(read(c.res('boot_check.json')))
        self.assertEqual(check['missing'], [])
        self.assertEqual(check['crlf'], [])
        self.assertEqual((check['secrets'], check['secret_modes']), (['key'], ['0o600']))
        self.assertEqual((check['env']['MODEL_ALIAS'], check['env']['CTX'], check['env']['MMPROJ']), (MODEL_ID, '524288', '1'))
        r2 = c.req()
        self.assertEqual((r2.status, r2.content()), (200, 'ok'))
        self.assertLess(r2.t_headers, 2.0)
        self.assertEqual(len(c.calls('new')), 1)
        self.assertNotIn(c.key, c.log())

    def test_02_nonstream_held(self):
        c = Case(self, FAKE_BOOT_S='3')
        c.serve()
        r = c.req(stream=False)
        self.assertEqual((r.status, r.content()), (200, 'ok'))
        self.assertGreaterEqual(r.t0 + r.t_headers, c.gate_at())
        self.assertEqual(r.comments, [])

    def test_02b_models_local_cors_and_single_model_routing(self):
        c = Case(self)
        c.serve()
        opt = c.req(method='OPTIONS', path='/v1/chat/completions', auth=False,
                    extra_headers={'Origin': 'http://localhost:3000', 'Access-Control-Request-Headers': 'Authorization, x-api-key'})
        self.assertEqual(opt.status, 204)
        self.assertEqual(opt.headers.get('Access-Control-Allow-Origin'), '*')
        self.assertIn('x-api-key', opt.headers.get('Access-Control-Allow-Headers', ''))
        r = c.req(method='GET', path='/v1/models', extra_headers={'Origin': 'http://localhost:3000'})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers.get('Access-Control-Allow-Origin'), '*')
        parsed = json.loads(r.raw)
        self.assertEqual([m['id'] for m in parsed['data']], [MODEL_ID])
        self.assertEqual([m['id'] for m in parsed['models']], [MODEL_ID])
        self.assertEqual((parsed['data'][0]['context_length'], parsed['data'][0]['image_input']), (262144, True))
        self.assertEqual(c.req(method='GET', path='/v1/models', auth=False).status, 401)
        self.assertEqual(c.req(host='evil.example:%d' % c.port).status, 403)
        self.assertEqual(c.calls('new'), [])
        # Single-model fallback routes claude-haiku-* / arbitrary names to MODEL_ID cleanly (also via x-api-key)
        msgs = [{'role': 'user', 'content': 'hi'}]
        resp = json.loads(c.req(path='/v1/messages', auth=False, extra_headers={'x-api-key': c.key},
                                body={'model': 'claude-haiku-4-5', 'messages': msgs}).raw)
        self.assertEqual(resp['model'], MODEL_ID)
        resp2 = json.loads(c.req(body={'model': 'gpt-5', 'messages': msgs, 'reasoning_effort': 'max'}).raw)
        self.assertEqual((resp2['model'], resp2['effort']), (MODEL_ID, 'max'))

    def test_03_boot_failure_multiprotocol_sse(self):
        c = Case(self, LLM_BOOT_ENV='FAKE_MODE=fail X_EXTRA=1', FAKE_BOOT_S='1')
        c.serve()
        out = {}
        specs = (('chat', '/v1/chat/completions', True),
                 ('anth', '/v1/messages', True),
                 ('resp', '/v1/responses', True),
                 ('json', '/v1/chat/completions', False))
        ts = [threading.Thread(target=lambda k=k, p=p, s=s: out.__setitem__(k, c.req(path=p, stream=s))) for k, p, s in specs]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        chat, anth, resp_sse, js = out['chat'], out['anth'], out['resp'], out['json']
        self.assertEqual(chat.status, 200)
        self.assertTrue(chat.comments)
        self.assertEqual(chat.data[-1], '[DONE]')
        err = json.loads(chat.data[-2])['error']
        self.assertEqual(err['code'], 503)
        self.assertIn('FAILED', err['message'])
        self.assertIn('event: error', anth.lines)
        anth_err = json.loads(anth.data[-1])
        self.assertEqual(anth_err['type'], 'error')
        self.assertIn('FAILED', anth_err['error']['message'])
        self.assertIn('event: error', resp_sse.lines)
        resp_err = json.loads(resp_sse.data[-1])
        self.assertEqual((resp_err['type'], resp_err['code']), ('error', 503))
        self.assertIn('FAILED', resp_err['message'])
        self.assertEqual(js.status, 503)
        self.assertIn('FAILED', json.loads(js.raw)['error']['message'])
        c.wait(lambda: c.sessions() == {} and c.session_json() is None, 15)
        c.assert_all_stopped(self)
        self.assertGreaterEqual(len(c.calls('stop')), 1)
        again = c.req(stream=False)
        self.assertEqual(again.status, 503)
        self.assertLess(again.t_headers, 1.0)
        self.assertEqual(len(c.calls('new')), 1)
        env = json.loads(read(c.res('boot_check.json')))['env']
        self.assertEqual((env['FAKE_MODE'], env['X_EXTRA'], env['CTX'], env['VRAM_RESERVE']), ('fail', '1', '524288', '700'))

    def test_04_idle_stop_and_restart(self):
        c = Case(self, LLM_IDLE_MIN='0.05')
        c.serve()
        r = c.req()
        self.assertEqual((r.status, r.content()), (200, 'ok'))
        t_end = time.time()
        c.wait(lambda: c.sessions() == {}, 20)
        self.assertGreater(time.time() - t_end, 2.5)
        self.assertIn('idle', c.log())
        c.assert_all_stopped(self)
        self.assertEqual(c.health_status(), 503)
        r = c.req(stream=True)
        self.assertEqual((r.status, r.content()), (200, 'ok'))
        self.assertEqual(len(c.calls('new')), 2)

    def test_05_concurrent(self):
        c = Case(self, FAKE_BOOT_S='2')
        c.serve()
        out = {}
        kinds = ('sse1', 'sse2', 'json')
        ts = [threading.Thread(target=lambda k=k: out.__setitem__(k, c.req(stream=k.startswith('sse')))) for k in kinds]
        [t.start() for t in ts]
        [t.join(60) for t in ts]
        self.assertEqual([(out[k].status, out[k].content()) for k in kinds], [(200, 'ok')] * 3)
        self.assertEqual(len(c.calls('new')), 1)
        ts = [threading.Thread(target=lambda k=k: out.__setitem__(k, c.req(stream=True))) for k in ('a', 'b')]
        [t.start() for t in ts]
        [t.join(30) for t in ts]
        self.assertEqual([(out[k].status, out[k].content(), out[k].comments) for k in 'ab'], [(200, 'ok', [])] * 2)

    def test_06_up_status_down_foreground(self):
        c = Case(self)
        p = c.serve()
        r = c.llm('up')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for line in ('Starting a Colab G4 GPU', 'G4 GPU allocated', 'Basalt v0.5.7 built', 'model downloaded (Basalt UD-IQ1_M)',
                     'model loaded onto GPU + RAM', 'tmpfs RAM punched & reclaimed', 'Engram table loaded (110.6 GB)',
                     'warmup answer done', 'Ready after 0:',
                     'Web UI     http://127.0.0.1:%d/' % c.port,
                     'Base URL   http://127.0.0.1:%d/v1' % c.port,
                     'API key    ' + c.key,
                     'Model      %s' % MODEL_ID):
            self.assertIn(line, r.stdout)
        self.assertNotIn('\x1b[', r.stdout)
        again = c.llm('up')
        self.assertIn('The GPU is already running', again.stdout)
        self.assertNotIn('Starting a Colab G4 GPU', again.stdout)
        self.assertEqual(c.llm('ask', 'hi').stdout.strip(), 'ok')
        st = c.llm('status')
        self.assertRegex(st.stdout, r'GPU +ready, idle 0:\d\d of 10 minutes, 1 request served, ran 0:\d\d, about 0\.\d\d compute units')
        self.assertIn('Telemetry  160.4 tok/s · hit 96% · pcie 0.8%', st.stdout)
        self.assertRegex(st.stdout, r'Colab +session %s running,' % c.name)
        r = c.llm('down')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('Stopped · the GPU ran 0:', r.stdout)
        self.assertIsNone(c.session_json())
        c.assert_all_stopped(self)
        self.assertIsNone(p.poll())
        self.assertEqual(c.health_status(), 503)

    def test_07_up_detached_then_down(self):
        c = Case(self, LLM_IMAGES='0', LLM_CONTEXT='131072')
        r = c.llm('up')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('131,072 tokens/slot (2 slots) · dynamic vision off · ~160.4 tok/s', r.stdout)
        lock = json.loads(read(os.path.join(c.state, 'serve.lock')))
        self.assertEqual((lock['detached'], lock['choice']), (True, [False, 131072]))
        self.assertEqual(json.loads(read(c.res('boot_check.json')))['env']['CTX'], '262144')
        c.env['LLM_IMAGES'] = '1'
        self.assertNotEqual(c.llm('up').returncode, 0)
        self.assertEqual(c.req().status, 200)
        r = c.llm('down')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        c.wait(lambda: not procs_with(c.dir, '--detached'), 10)
        self.assertFalse(os.path.exists(os.path.join(c.state, 'serve.lock')))
        c.assert_all_stopped(self)

    def test_07b_launchers_env_and_ask_image(self):
        c = Case(self, LLM_IMAGES='1', LLM_CONTEXT='131072')
        for tool in ('opencode', 'claude', 'codex'):
            self.assertIn('not running', c.llm(tool).stderr)
        fake = os.path.join(c.dir, 'bin')
        os.makedirs(fake)
        with open(os.path.join(fake, 'opencode'), 'w') as f:
            f.write('#!/bin/sh\necho "config=$OPENCODE_CONFIG args=$*"\n')
        with open(os.path.join(fake, 'claude'), 'w') as f:
            f.write('#!/bin/sh\necho "url=$ANTHROPIC_BASE_URL key=$ANTHROPIC_API_KEY model=$ANTHROPIC_MODEL args=$*"\n')
        with open(os.path.join(fake, 'codex'), 'w') as f:
            f.write('#!/bin/sh\necho "url=$OPENAI_BASE_URL key=$OPENAI_API_KEY model=$OPENAI_MODEL args=$*"\n')
        for tool in ('opencode', 'claude', 'codex'):
            os.chmod(os.path.join(fake, tool), 0o755)
        c.env['PATH'] = fake + os.pathsep + c.env['PATH']
        self.assertEqual(c.llm('up').returncode, 0)
        r_oc = c.llm('opencode', '/some/project')
        self.assertEqual(r_oc.stdout.strip(), 'config=%s args=/some/project' % os.path.join(c.state, 'opencode.json'))
        cfg = json.loads(read(os.path.join(c.state, 'opencode.json')))['provider']['ds41']
        self.assertEqual((cfg['options']['baseURL'], cfg['models'][MODEL_ID]['limit']['context']), ('http://127.0.0.1:%d/v1' % c.port, 131072))
        r_cl = c.llm('claude', '-p', 'hi')
        self.assertEqual(r_cl.stdout.strip(), 'url=http://127.0.0.1:%d key=%s model=%s args=-p hi' % (c.port, c.key, MODEL_ID))
        r_cx = c.llm('codex', 'exec', 'test')
        self.assertEqual(r_cx.stdout.strip(), 'url=http://127.0.0.1:%d/v1 key=%s model=%s args=exec test' % (c.port, c.key, MODEL_ID))
        r_env = c.llm('env')
        self.assertIn("export OPENAI_BASE_URL=", r_env.stdout)
        self.assertIn("export ANTHROPIC_MODEL=", r_env.stdout)
        r_ps = c.llm('env', '--powershell')
        self.assertIn("$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:%d'" % c.port, r_ps.stdout)
        img = os.path.join(c.dir, 'dot.png')
        with open(img, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\nfake')
        r_ask = c.llm('ask', '--max', '--image', img, 'what is this', input='extra context')
        self.assertEqual((r_ask.returncode, r_ask.stdout.strip()), (0, 'ok'))
        self.assertEqual(c.llm('down').returncode, 0)
        c.wait(lambda: not procs_with(c.dir, '--detached'), 10)
        c.assert_all_stopped(self)

    def test_07c_saved_settings_and_hot_reload(self):
        c = Case(self)
        del c.env['LLM_IDLE_MIN'], c.env['LLM_MAX_HOURS']
        with open(os.path.join(c.state, 'settings.json'), 'w') as f:
            json.dump({'images': False, 'context': 131072, 'effort': 'high', 'idle_min': 7, 'max_hours': 2, 'uncensor': False}, f)
        r = c.llm('up')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('%s (effort high)' % MODEL_ID, r.stdout)
        self.assertIn('stops after 7 minutes without requests', r.stdout)
        env = json.loads(read(c.res('boot_check.json')))['env']
        self.assertEqual((env['CTX'], env['UNCENSOR']), ('262144', '0'))
        msgs = [{'role': 'user', 'content': 'hi'}]
        self.assertEqual(json.loads(c.req(body={'model': MODEL_ID, 'messages': msgs}).raw)['effort'], 'high')
        # Hot-reload effort & idle_min live without restarting G4
        rl = c.req(path='/_llm/reload', body={'effort': 'max', 'idle_min': 15, 'max_hours': 4})
        self.assertEqual(rl.status, 200)
        rl_data = json.loads(rl.raw)
        self.assertEqual((rl_data['applied']['effort'], rl_data['applied']['idle_min'], rl_data['restart_required']), ('max', 15.0, False))
        self.assertEqual(json.loads(c.req(body={'model': MODEL_ID, 'messages': msgs}).raw)['effort'], 'max')
        self.assertEqual(json.loads(c.req(path='/v1/responses', body={'model': MODEL_ID, 'input': 'hi'}).raw)['effort'], 'max')
        self.assertIn('hard stop 2 h', c.llm('settings').stderr)
        self.assertEqual(c.llm('down').returncode, 0)
        c.wait(lambda: not procs_with(c.dir, '--detached'), 10)
        c.assert_all_stopped(self)

    def test_07d_alloc_retry_and_orphan_boot_cancel(self):
        # 1) First G4 allocation fails -> automatic backoff retry succeeds on attempt 2
        c = Case(self, FAKE_ALLOC_FAILS='1')
        c.serve()
        r = c.req()
        self.assertEqual((r.status, r.content()), (200, 'ok'))
        self.assertEqual(len(c.calls('new')), 2)
        c.cleanup()
        # 2) Request-triggered cold boot where sole SSE waiter disconnects -> auto-cancels after LLM_ORPHAN_BOOT_S
        c2 = Case(self, FAKE_BOOT_S='10', LLM_ORPHAN_BOOT_S='1.0')
        c2.serve()
        conn = http.client.HTTPConnection('127.0.0.1', c2.port, timeout=10)
        conn.request('POST', '/v1/chat/completions',
                     body=json.dumps({'model': MODEL_ID, 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True}).encode(),
                     headers={'Authorization': 'Bearer ' + c2.key, 'Content-Type': 'application/json'})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        resp.read1(128)  # receive initial heartbeat
        c2.wait(lambda: c2.calls('new'), 10)  # wait until cold boot has actually started
        conn.close()  # client hangs up during cold boot
        c2.wait(lambda: 'all waiting clients disconnected' in c2.log(), 15)
        c2.wait(lambda: c2.sessions() == {} and c2.session_json() is None, 15)
        c2.assert_all_stopped(self)

    def test_08_lock_blocks_second_serve(self):
        c = Case(self)
        p = c.serve()
        r = c.llm('serve', timeout=20)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('already running', r.stderr)
        os.kill(p.pid, signal.SIGKILL)
        p.wait(10)
        self.assertTrue(os.path.exists(os.path.join(c.state, 'serve.lock')))
        c.serve()
        self.assertEqual(c.health_status(), 503)

    def test_09_watchdog_stops_vm_when_proxy_dies(self):
        c = Case(self, LLM_IDLE_MIN='0.05')
        p = c.serve()
        self.assertEqual(c.req().status, 200)
        os.kill(p.pid, signal.SIGKILL)
        t_kill = time.time()
        c.wait(lambda: c.sessions() == {}, 20)
        self.assertGreater(time.time() - t_kill, 2.5)
        self.assertIn('proxy gone', c.log())
        r = c.llm('down')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsNone(c.session_json())
        c.assert_all_stopped(self)

    def test_10_watchdog_hard_cap(self):
        c = Case(self, LLM_MAX_HOURS='0.0025')
        p = c.serve()
        self.assertEqual(c.req().status, 200)
        t0 = c.session_json()['t0']
        os.kill(p.pid, signal.SIGKILL)
        c.wait(lambda: c.sessions() == {}, 25)
        self.assertGreater(time.time(), t0 + 9)
        self.assertIn('watchdog %s: hard stop' % c.name, c.log())
        self.assertEqual(c.llm('down').returncode, 0)
        c.assert_all_stopped(self)

    def test_11_proxy_hard_cap(self):
        c = Case(self, LLM_MAX_HOURS='0.0025')
        p = c.serve()
        self.assertEqual(c.req().status, 200)
        c.wait(lambda: c.sessions() == {}, 25)
        c.wait(lambda: 'stopping the VM (hard stop' in c.log(), 10)
        c.wait(lambda: c.session_json() is None, 10)
        c.assert_all_stopped(self)
        self.assertIsNone(p.poll())

    def test_12_vm_lost_no_tunnel_respawn(self):
        c = Case(self)
        c.serve()
        self.assertEqual(c.req().status, 200)
        r = subprocess.run(FAKE_COLAB.split() + ['stop', '-s', c.name], env=c.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0)
        c.wait(lambda: c.session_json() is None, 20)
        c.assert_all_stopped(self)
        self.assertEqual(len(c.lines('ssh_calls.log')), 1)
        r = c.req(stream=True)
        self.assertEqual((r.status, r.content()), (200, 'ok'))
        self.assertEqual(len(c.calls('new')), 2)

    def test_14_request_during_boot_held_until_gate(self):
        c = Case(self, FAKE_BOOT_S='4')
        c.serve()
        out = {}
        t = threading.Thread(target=lambda: out.__setitem__('sse', c.req(stream=True)))
        t.start()
        c.wait(lambda: c.lines('ssh_calls.log'), 15)
        c.wait(lambda: [f for f in c.lines('fwd.log') if not f['ok']], 15)
        self.assertFalse(os.path.exists(c.res('gate_open_at')))
        js = c.req(stream=False)
        t.join(60)
        gate, sse = c.gate_at(), out['sse']
        self.assertEqual((sse.status, sse.content(), js.status, js.content()), (200, 'ok', 200, 'ok'))
        first_data = min(t0 for t0, d in sse.chunks if b'data: ' in d)
        self.assertGreaterEqual(sse.t0 + first_data, gate)
        self.assertGreaterEqual(js.t0 + js.t_headers, gate)
        self.assertGreaterEqual(len(c.assert_no_forward_before_gate(self)), 3)
        self.assertNotIn('open failed', c.log())

    def test_15_hf_token_reaches_the_vm_only(self):
        c = Case(self, HF_TOKEN=HF_SECRET)
        c.serve()
        self.assertEqual(c.req().status, 200)
        check = json.loads(read(c.res('boot_check.json')))
        self.assertEqual((check['secrets'], check['secret_modes']), (['hf_hdr', 'key'], ['0o600']))
        self.assertNotIn(HF_SECRET, c.log())

    def test_13_sigterm_stops_vm(self):
        c = Case(self)
        p = c.serve()
        self.assertEqual(c.req().status, 200)
        p.send_signal(signal.SIGTERM)
        p.wait(20)
        self.assertEqual(c.sessions(), {})
        self.assertIsNone(c.session_json())
        c.assert_all_stopped(self)
        self.assertFalse(os.path.exists(os.path.join(c.state, 'serve.lock')))


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix='unit-', dir=TMP)
        state = os.path.join(cls.dir, 'state')
        shutil.copytree(BASE_STATE, state)
        old = dict(os.environ)
        os.environ.update(LLM_STATE=state, LLM_VM_DIR=VM_DIR, LLM_COLAB=FAKE_COLAB, LLM_SESSION='unit')
        os.environ.pop('LLM_SSH_CMD', None)
        try:
            sys.path.insert(0, ROOT)
            cls.llm = importlib.import_module('llm')
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_setup_outputs(self):
        L = self.llm
        self.assertRegex(read(L.KEY_FILE), r'^sk-ds41-[0-9a-f]{48}\n$')
        self.assertEqual(os.stat(L.KEY_FILE).st_mode & 0o777, 0o600)
        self.assertTrue(os.path.exists(L.SSH_KEY))
        cfg = json.loads(read(os.path.join(L.STATE, 'opencode.json')))['provider']['ds41']
        self.assertEqual(cfg['options']['baseURL'], 'http://127.0.0.1:18080/v1')
        self.assertEqual((cfg['options']['headerTimeout'], cfg['options']['chunkTimeout']), (1800000, 900000))
        self.assertEqual(cfg['models'][MODEL_ID]['limit']['context'], 262144)
        self.assertEqual(cfg['models'][MODEL_ID]['modalities']['input'], ['text', 'image'])

    def test_vm_files_in_sync(self):
        self.assertEqual(self.llm.missing_vm_files(), [])

    def test_bundle_lf_and_modes(self):
        L = self.llm
        vm = os.path.join(self.dir, 'vmcrlf')
        files = L.vm_files()
        for f in files:
            os.makedirs(os.path.dirname(os.path.join(vm, f)), exist_ok=True)
            with open(os.path.join(VM_DIR, f), 'rb') as a, open(os.path.join(vm, f), 'wb') as b:
                b.write(a.read().replace(b'\n', b'\r\n'))
        old, L.VM_DIR = L.VM_DIR, vm
        try:
            path = os.path.join(self.dir, 'b.tgz')
            L.build_bundle(path)
            with tarfile.open(path) as t:
                names = {m.name: m.mode for m in t.getmembers()}
                body = {m.name: t.extractfile(m).read() for m in t.getmembers()}
        finally:
            L.VM_DIR = old
        self.assertEqual(sorted(names), sorted(['vm/' + f for f in files] + ['secrets/key']))
        self.assertIn('vm/patch_basalt.py', names)
        self.assertIn('vm/basalt_jobs.py', names)
        self.assertNotIn('vm/patches/series', names)
        self.assertFalse([n for n, b in body.items() if b'\r' in b])
        self.assertEqual({names[n] for n in names if not n.startswith('vm/')}, {0o600})
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertEqual(body['vm/boot.sh'], read(os.path.join(VM_DIR, 'boot.sh')).encode())

    def test_images_and_context_boot_env(self):
        L = self.llm
        cases = {('0', '262144'): ("MODEL_ALIAS='%s'" % MODEL_ID, "CTX='524288'", "MMPROJ='0'", "VRAM_RESERVE='700'", "RAM_BUDGET='55'"),
                 ('1', '64K'): ("MODEL_ALIAS='%s'" % MODEL_ID, "CTX='131072'", "MMPROJ='1'", "VRAM_RESERVE='700'"),
                 ('0', '32k'): ("CTX='65536'", "MMPROJ='0'", "VRAM_RESERVE='700'"),
                 ('0', '131072'): ("CTX='262144'", "MMPROJ='0'", "VRAM_RESERVE='700'")}
        try:
            for (images, context), want in cases.items():
                os.environ['LLM_IMAGES'], os.environ['LLM_CONTEXT'] = images, context
                envs = L.boot_env()
                for w in want:
                    self.assertIn(w, envs, (images, context))
        finally:
            os.environ.pop('LLM_IMAGES', None)
            os.environ.pop('LLM_CONTEXT', None)
        self.assertEqual([L.context_value(v) for v in ('300000', '5000', '262K', 'abc', '4095')], [None, 5120, 262144, None, None])

    def test_boot_env(self):
        L = self.llm
        os.environ['LLM_BOOT_ENV'] = 'DRAFT=0 X_A=1'
        try:
            envs = L.boot_env()
            self.assertIn("MODEL_ALIAS='%s'" % MODEL_ID, envs)
            self.assertIn("CTX='524288'", envs)
            self.assertIn("MMPROJ='1'", envs)
            self.assertIn("DRAFT='0' X_A='1'", envs)
            self.assertIn("(%s setsid nohup bash vm/boot.sh" % envs, L.boot_cmd(envs))
            self.assertNotIn('libpin.so', L.boot_cmd(envs))
            os.environ['LLM_BOOT_ENV'] = "A=it's"
            self.assertRaises(L.StartError, L.boot_env)
            os.environ['LLM_BOOT_ENV'] = 'NOEQUALS'
            self.assertRaises(L.StartError, L.boot_env)
        finally:
            del os.environ['LLM_BOOT_ENV']

    def test_ssh_cmd_windows_quoting(self):
        L = self.llm
        old = L.COLAB, L.SSH_KEY
        L.COLAB = [r'C:\Users\Jo Doe\.local\bin\colab.exe', '--auth', 'oauth2']
        L.SSH_KEY = r'C:\Users\Jo Doe\llm 100%\state\id_ed25519'
        try:
            cmd = L.ssh_cmd(28999, win=True)
        finally:
            L.COLAB, L.SSH_KEY = old
        self.assertEqual(cmd[cmd.index('-F') + 1], 'NUL')
        self.assertIn('UserKnownHostsFile=NUL', cmd)
        self.assertEqual(cmd[cmd.index('-i') + 1], r'C:\Users\Jo Doe\llm 100%\state\id_ed25519')
        self.assertEqual(cmd[cmd.index('-L') + 1], '127.0.0.1:28999:127.0.0.1:18080')
        self.assertIn(r'ProxyCommand="C:\Users\Jo Doe\.local\bin\colab.exe" --auth oauth2 ssh --proxy-mode -s unit -i '
                      r'"C:\Users\Jo Doe\llm 100%%\state\id_ed25519"', cmd)
        self.assertEqual(cmd[-1], 'unit')

    def test_split_cmd(self):
        L = self.llm
        self.assertEqual(L.split_cmd(r'"C:\Program Files\Py\python.exe" C:\t\fake.py', win=True),
                         [r'C:\Program Files\Py\python.exe', r'C:\t\fake.py'])
        self.assertEqual(L.split_cmd("python3 '/a b/c.py'", win=False), ['python3', '/a b/c.py'])

    def test_proc_identity(self):
        L = self.llm
        self.assertTrue(L.alive(L.me()))
        self.assertFalse(L.alive({'pid': os.getpid(), 'start': 'other'}))
        p = subprocess.Popen(['true'])
        p.wait()
        self.assertIsNone(L.proc_start(p.pid))
        self.assertFalse(L.alive({'pid': p.pid, 'start': None}))

    def test_lock_stale_and_busy(self):
        L = self.llm
        self.assertIsNone(L.take_lock({'port': 1}))
        self.assertEqual(L.take_lock({'port': 2})['pid'], os.getpid())
        L.release_lock()
        self.assertFalse(os.path.exists(L.LOCK))
        p = subprocess.Popen(['true'])
        p.wait()
        with open(L.LOCK, 'w') as f:
            json.dump({'pid': p.pid, 'start': '1', 'port': 9}, f)
        self.assertIsNone(L.take_lock({'port': 3}))
        L.release_lock()
        open(L.LOCK, 'w').close()
        self.assertTrue(L.take_lock({'port': 4})['busy'])
        os.utime(L.LOCK, (time.time() - 60, time.time() - 60))
        self.assertIsNone(L.take_lock({'port': 5}))
        L.release_lock()

    def test_log_redaction_and_rotation(self):
        L = self.llm
        key = L.api_key()
        L.log('leak %s and Authorization: Bearer abc123' % key, echo=False)
        text = read(L.LOG)
        self.assertNotIn(key, text)
        self.assertNotIn('abc123', text)
        with open(L.LOG, 'a') as f:
            f.write('x' * 1100000)
        L.log('after rotation', echo=False)
        self.assertTrue(os.path.exists(L.LOG + '.1'))
        self.assertLess(os.path.getsize(L.LOG), 1000)

    def test_auto_create_detection(self):
        L = self.llm
        p = subprocess.Popen([PY, '-c', "import sys; sys.stderr.write(\"[colab] Creating runtime 'unit'...\\n\")"], stderr=subprocess.PIPE)
        hits = []
        L.drain(p, lambda: hits.append(1))
        p.wait()
        p.stderr.close()
        self.assertEqual(hits, [1])

    def test_basalt_native_rewrite_config_jobs_and_abliterate(self):
        import re, struct, zipfile
        L = self.llm
        old_effort = L.SETTINGS['effort']
        try:
            L.SETTINGS['effort'] = 'high'
            chat = json.loads(L.rewrite_request(json.dumps({'model': 'claude-sonnet-4-5', 'messages': []}).encode()))
            self.assertEqual((chat['model'], chat['reasoning_effort']), (MODEL_ID, 'high'))
            resp_req = json.loads(L.rewrite_request(json.dumps({'model': 'gpt-5', 'input': 'hi'}).encode(), '/v1/responses'))
            self.assertEqual((resp_req['model'], resp_req['reasoning']), (MODEL_ID, {'effort': 'high'}))
            self.assertNotIn('reasoning_effort', resp_req)
            anth_req = json.loads(L.rewrite_request(json.dumps({'model': 'claude-haiku-4-5', 'messages': []}).encode(), '/v1/messages'))
            self.assertEqual(anth_req['model'], MODEL_ID)
            self.assertNotIn('reasoning_effort', anth_req)
            os.environ['LLM_STRICT_MODEL'] = '1'
            strict_req = json.loads(L.rewrite_request(json.dumps({'model': 'other-model', 'messages': []}).encode()))
            self.assertEqual(strict_req['model'], 'other-model')
        finally:
            L.SETTINGS['effort'] = old_effort
            os.environ.pop('LLM_STRICT_MODEL', None)

        self.assertEqual(L.parse_ask_args(['--max', '--image', 'a.png', 'hello', 'world']), ('max', ['a.png'], 'hello world'))
        self.assertEqual(L.parse_ask_args(['--effort', '75', 'hi']), ('75', [], 'hi'))
        self.assertIn('160.4 tok/s', L.speed_note(True, 262144))

        # Verify vm/serve.sh v2 config schema (answer-room & idle-unload in models[0], conversation cache in engine)
        serve_sh = read(os.path.join(VM_DIR, 'serve.sh'))
        m = re.search(r"python3 - \"\$R/basalt_config\.json\" <<'EOF'\n(.*?)\nEOF", serve_sh, re.S)
        self.assertIsNotNone(m)
        cfg_path = os.path.join(self.dir, 'basalt_config.json')
        env = dict(os.environ, PORT='18081', CTX='524288', MMPROJ='1', DRAFT='1', VRAM_RESERVE='700',
                   RAM_BUDGET='55', DRAFT_EXPERTS='32', SPEC_MAX='6', ANSWER_ROOM='0',
                   MODEL_ALIAS=MODEL_ID, CONV_CACHE_MIB='32768', CONV_CACHE_DIR='/content/conv_cache',
                   MODEL_PATH='/mnt/ram/model.basalt', ENGRAM_PATH='/mnt/ram/engram.bin')
        r = subprocess.run([PY, '-c', m.group(1), cfg_path], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        cfg = json.loads(read(cfg_path))
        self.assertEqual(cfg['version'], 2)
        self.assertNotIn('answer-room', cfg['server'])
        self.assertNotIn('idle-unload', cfg['server'])
        self.assertEqual((cfg['models'][0]['alias'], cfg['models'][0]['answer-room'], cfg['models'][0]['idle-unload']), (MODEL_ID, 0, 0))
        self.assertEqual((cfg['engine']['conversation-cache-disk-mib'], cfg['engine']['conversation-cache-dir']), (32768, '/content/conv_cache'))

        # Verify vm/basalt_jobs.py + punch_sections + vm/abliterate_basalt.py against a synthetic BASALTMF v2 pack
        sys.path.insert(0, VM_DIR)
        try:
            bj = importlib.import_module('basalt_jobs')
            ab = importlib.import_module('abliterate_basalt')
            hdr = bytearray(65536)
            sections = [
                ('meta', 65536, 4096),
                ('gguf', 131072, 1048576),
                ('dspark/gguf', 1179648, 262144),
                ('vision/mmproj', 1441792, 65536),
                ('experts', 2097152, 4194304),
                ('engram', 8388608, 16777216),
            ]
            file_bytes = 8388608 + 16777216
            struct.pack_into('<8sIIQ', hdr, 0, b'BASALTMF', 2, len(sections), file_bytes)
            for i, (name, soff, sbytes) in enumerate(sections):
                struct.pack_into('<64sQQ', hdr, 64 + i * 128, name.encode(), soff, sbytes)
            fb, parsed = bj.parse_header(bytes(hdr))
            self.assertEqual(fb, file_bytes)
            bb_jobs, eng_jobs = bj.build_jobs('https://example.test/m.basalt', fb, parsed, '/mnt/ram/model.basalt', '/mnt/ram/engram.bin')
            self.assertEqual(len(eng_jobs), 1)
            self.assertEqual((eng_jobs[0]['src_off'], eng_jobs[0]['dest_off'], eng_jobs[0]['size']), (8388608, 0, 16777216))
            self.assertEqual(sum(j['size'] for j in bb_jobs), 65536 + 4096 + 1048576 + 262144 + 65536 + 4194304)

            if sys.platform.startswith('linux'):
                pack_file = os.path.join(self.dir, 'punch_test.basalt')
                with open(pack_file, 'wb') as f:
                    f.write(hdr)
                    f.seek(1179648)
                    f.write(b'\xAA' * 262144)
                    f.seek(1441792)
                    f.write(b'\x55' * 65536)
                blocks_before = os.stat(pack_file).st_blocks
                freed = bj.punch_sections(pack_file, {'dspark/gguf'})
                blocks_after = os.stat(pack_file).st_blocks
                self.assertEqual(freed, 262144)
                self.assertLess(blocks_after, blocks_before)
                with open(pack_file, 'rb') as f:
                    f.seek(1179648)
                    self.assertEqual(f.read(4096), b'\x00' * 4096)
                    f.seek(1441792)
                    self.assertEqual(f.read(4096), b'\x55' * 4096)

            dim = 4
            def npy(shape, fortran, vals):
                h = ("{'descr': '<f4', 'fortran_order': %s, 'shape': %r, }" % (fortran, shape)).encode()
                h = h + b' ' * ((64 - (10 + len(h) + 1) % 64) % 64) + b'\n'
                return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(h)) + h + struct.pack('<%df' % len(vals), *vals)
            npz_path = os.path.join(self.dir, 'b_edit.npz')
            with zipfile.ZipFile(npz_path, 'w') as zf:
                zf.writestr('Q.npy', npy((dim, 2), True, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
                zf.writestr('scales.npy', npy((2, 2), False, [0.5, 0.0, 0.0, 0.75]))
            _, secs = ab.parse_basalt_header(bytes(hdr))
            meta = {'tensors': {
                'blk.0.attn_output_b.weight': {'section': 'gguf', 'offset': 0, 'bytes': 210 * dim, 'shape': [256, dim], 'type_id': 14},
                'blk.1.ffn_down_shexp.weight': {'section': 'gguf', 'offset': 4096, 'bytes': 176 * dim, 'shape': [256, dim], 'type_id': 13},
            }}
            _, _, _, plan = ab.build_plan_from_meta('/mnt/ram/model.basalt', secs, meta, npz_path)
            self.assertEqual([(p[1], p[2], p[5], p[7]) for p in plan],
                             [('blk.0.attn_output_b.weight', 14, 131072, 0.5),
                              ('blk.1.ffn_down_shexp.weight', 13, 131072 + 4096, 0.75)])
        finally:
            sys.path.remove(VM_DIR)


if __name__ == '__main__':
    unittest.main()
