import asyncio
from asyncio import Queue
import threading

import sh
from tornado import ioloop


def tail(filename, msg, pid):
    for line in sh.tail("-f", "--pid", pid, filename, _iter=True):
        print(f'{msg}: {line}', end='')


class TailManager:
    threads = Queue()
    _started = False

    @classmethod
    def start_tail(cls, file_path, msg, pid):
        thread = threading.Thread(target=tail, args=(file_path, msg, pid))
        cls.start_thread(thread)
        cls.start_clean_inactivate()

    @classmethod
    def start_thread(cls, thread):
        thread.start()
        ioloop.IOLoop.current().add_callback(cls.threads.put, thread)

    @classmethod
    def start_clean_inactivate(cls):
        if cls._started:
            return
        ioloop.IOLoop.current().add_callback(cls.clean_inactivate)
        cls._started = True

    @classmethod
    async def clean_inactivate(cls, interval=2):
        while True:
            for _ in range(cls.threads.qsize()):
                t: threading.Thread = await cls.threads.get()
                if t.is_alive():
                    await cls.threads.put(t)
                else:
                    t.join()
            await asyncio.sleep(interval)


class LogRecorder(object):
    pid_log_path = dict()

    @classmethod
    def record_pid_log_path(cls, pid, file_path):
        cls.pid_log_path[pid] = file_path

    @classmethod
    def get_log_from_pid(cls, pid):
        file_path = cls.pid_log_path.get(pid)
        if file_path:
            with open(file_path) as f:
                return f.read()
        else:
            return 'No log file found'
