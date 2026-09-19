"""Worker lifecycle regressions without OpenCV, OpenVINO or a live jobs DB."""
import importlib.util
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock


WORKER_PATH = Path(__file__).resolve().parents[1] / 'app' / 'worker' / 'main.py'


class WorkerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deps = mock.patch.dict(sys.modules, {
            'cv2': types.ModuleType('cv2'),
            'app.common.config': types.ModuleType('app.common.config'),
            'app.common.db': types.ModuleType('app.common.db'),
            'app.common.jobs': types.ModuleType('app.common.jobs'),
            'app.inference': types.ModuleType('app.inference'),
        })
        cls.deps.start()
        config = sys.modules['app.common.config']
        for name in ('GFPGAN_MODEL_PATH', 'INPUT_DIR', 'MODEL_PATH', 'OUTPUT_DIR', 'TMP_DIR'):
            setattr(config, name, Path('/unused'))
        config.MODEL_REGISTRY = {}
        config.ensure_directories = lambda: None
        config.face_models_available = lambda: False
        jobs = sys.modules['app.common.jobs']
        jobs.QUALITY_VALUES = {}
        jobs.crop_box = lambda *args: None
        db = sys.modules['app.common.db']
        db.connect = lambda: None  # overridden for each isolated test
        db.init_db = lambda: None
        db.utc_now = lambda: '2026-09-19T00:00:00+00:00'
        sys.modules['app.inference'].OpenVINORealESRGANer = object
        spec = importlib.util.spec_from_file_location('worker_lifecycle_under_test', WORKER_PATH)
        cls.worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.worker)

    @classmethod
    def tearDownClass(cls):
        cls.deps.stop()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.db_path = root / 'test.db'
        self.connections = []
        self.worker.TMP_DIR = root
        self.worker.OUTPUT_DIR = root
        self.worker.INPUT_DIR = root
        self.job = {'id': 'job-1', 'input_path': 'source.png', 'output_path': 'result.png',
                    'height': 10, 'width': 10}

        def connect():
            connection = sqlite3.connect(self.db_path, timeout=3)
            connection.row_factory = sqlite3.Row
            self.connections.append(connection)
            return connection

        self.worker.connect = connect
        with connect() as connection:
            connection.executescript('''
                CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT, lease_owner TEXT,
                    lease_expires_at TEXT, heartbeat_at TEXT, progress INT, error TEXT,
                    finished_at TEXT);
                CREATE TABLE service_state (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
                CREATE TABLE job_metrics (job_id TEXT, worker_slot INT, decode_ms REAL,
                    sr_ms REAL, resize_ms REAL, face_ms REAL, encode_ms REAL,
                    verify_ms REAL, total_ms REAL, output_bytes INT, concurrent_jobs INT,
                    created_at TEXT);
                INSERT INTO jobs(id,status,lease_owner,progress)
                    VALUES ('job-1','processing','owner',0);
            ''')

    def tearDown(self):
        for connection in self.connections:
            connection.close()

    def row(self):
        with self.worker.connect() as connection:
            return dict(connection.execute('SELECT * FROM jobs WHERE id=?', ('job-1',)).fetchone())

    def cancel(self):
        with self.worker.connect() as connection:
            connection.execute("UPDATE jobs SET status='cancelled', lease_owner=NULL WHERE id='job-1'")

    def test_cancelled_job_cannot_be_failed_or_published(self):
        partial = self.worker.TMP_DIR / 'job-1.owner.partial.png'
        partial.write_bytes(b'partial')
        output = self.worker.OUTPUT_DIR / 'result.png'
        output.write_bytes(b'existing')
        self.cancel()
        self.assertFalse(self.worker.finish_job('job-1', 'owner', 'failed', 'cancelled'))
        self.assertFalse(self.worker.publish_success(self.job, 'owner', partial, None, 1))
        self.assertEqual(self.row()['status'], 'cancelled')
        self.assertEqual(output.read_bytes(), b'existing')
        self.assertFalse(partial.exists())

    def test_cancel_during_inference_cleans_partial_without_failing(self):
        partial = self.worker.TMP_DIR / 'job-1.owner.partial.png'

        def process(*args):
            partial.write_bytes(b'partial')
            self.cancel()
            raise self.worker.JobLeaseLost()

        with mock.patch.object(self.worker, 'process_job', side_effect=process), \
             mock.patch.object(self.worker, 'publish_success') as publish:
            self.worker.run_claimed_job(self.job, 'owner', 1, 1, OrderedDict({('general', 256): object()}), None)
        self.assertEqual(self.row()['status'], 'cancelled')
        self.assertFalse(partial.exists())
        publish.assert_not_called()

    def test_lease_lost_after_inference_does_not_publish_or_fail(self):
        partial = self.worker.TMP_DIR / 'job-1.owner.partial.png'
        partial.write_bytes(b'partial')

        def process(*args):
            args[-1].set()  # simulated failed renewal by the heartbeat
            return partial, {'total_ms': 1}

        with mock.patch.object(self.worker, 'process_job', side_effect=process), \
             mock.patch.object(self.worker, 'publish_success') as publish:
            self.worker.run_claimed_job(self.job, 'owner', 1, 1, OrderedDict({('general', 256): object()}), None)
        self.assertEqual(self.row()['status'], 'processing')
        self.assertFalse(partial.exists())
        publish.assert_not_called()

    def test_real_processing_error_marks_owned_job_failed_and_cleans_partial(self):
        partial = self.worker.TMP_DIR / 'job-1.owner.partial.png'

        def process(*args):
            partial.write_bytes(b'partial')
            raise ValueError('decode failed')

        with mock.patch.object(self.worker, 'process_job', side_effect=process):
            self.worker.run_claimed_job(self.job, 'owner', 1, 1, OrderedDict({('general', 256): object()}), None)
        self.assertEqual(self.row()['status'], 'failed')
        self.assertEqual(self.row()['error'], 'decode failed')
        self.assertFalse(partial.exists())

    def test_heartbeat_signals_lost_lease_on_rejection_or_exception(self):
        class ImmediateTick:
            def wait(self, delay):
                return False

        for result in (False, RuntimeError('db busy')):
            lost = threading.Event()
            with self.subTest(result=result), mock.patch.object(
                self.worker, 'renew_lease', side_effect=result if isinstance(result, Exception) else None,
                return_value=result if not isinstance(result, Exception) else None,
            ):
                self.worker.heartbeat_loop(ImmediateTick(), lost, 'job-1', 'owner')
                self.assertTrue(lost.is_set())

    def test_progress_check_aborts_cancelled_job(self):
        class Image:
            shape = (10, 10, 3)
            ndim = 3
        cv2 = sys.modules['cv2']
        cv2.IMREAD_UNCHANGED = 1
        with mock.patch.object(cv2, 'imread', return_value=Image(), create=True):
            self.cancel()
            with self.assertRaises(self.worker.JobLeaseLost):
                self.worker.process_job(object(), None, self.job, 'owner', threading.Event())


if __name__ == '__main__':
    unittest.main()
