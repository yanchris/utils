import os, sys
from twisted.internet import reactor, threads, defer
from twisted.logger import Logger
from twisted.logger import globalLogPublisher, textFileLogObserver, FilteringLogObserver, LogLevelFilterPredicate, LogLevel
from twisted.web.client import Agent, readBody
from twisted.web.http_headers import Headers
from datetime import datetime

import traceback
from inspect import currentframe

to_bytes = lambda x: bytes(x, 'utf8')
to_str = lambda x: x.decode('utf8')

def to_timestamp(x):
    try:
        return int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
    except:
        return int(datetime.strptime(x, "%Y-%m-%d").timestamp())

now_timestamp = lambda : int(datetime.utcnow().timestamp())

_logger = Logger()

class QLogger(object):
    def __init__(self, logPath):
        self.startLogging(logPath)

    @classmethod
    def startLogging(cls, logOutput, levelStr='debug'):
        if isinstance(logOutput, str):
            dir = os.path.dirname(logOutput)
            if dir and not os.path.exists(dir):
                os.makedirs(dir)
            logOutput = open(logOutput, 'a+')

        level = LogLevel.levelWithName(levelStr)
        predicate = LogLevelFilterPredicate(defaultLogLevel=level)
        observer = FilteringLogObserver(textFileLogObserver(outFile=logOutput), [predicate])
        globalLogPublisher.addObserver(observer)


def log_info(*kargv):
    ln = currentframe().f_back.f_lineno
    _logger.info('[line@{}]'.format(ln) + ' - ' + ''.join([str(msg) for msg in kargv]).replace('{','{{').replace('}','}}'))

def log_debug(*kargv):
    ln = currentframe().f_back.f_lineno
    _logger.debug('[line@{}]'.format(ln) + ' - ' + ''.join([str(msg) for msg in kargv]).replace('{','{{').replace('}','}}'))

def log_error(*kargv):
    ln = currentframe().f_back.f_lineno
    _logger.error('[line@{}]'.format(ln) + ' - ' + ''.join([str(msg) for msg in kargv]).replace('{','{{').replace('}','}}'))


def http_request(url, cb, *args, **kwargs):
    def request_failed(e):
        log_error('Request failed with {} for URL {}'.format(e, url))
        return e

    def check_request(response):
        log_info('URL:' + url)
        log_info('Response version: ',  response.version)
        log_info('Response code: ',  response.code)
        log_info('Response phrase: ', response.phrase)
        #log_info('Response headers: ')
        #log_info(pformat(list(response.headers.getAllRawHeaders())))

        if response.code == 200:
            r = readBody(response)
            r.addCallback(cb, *args, **kwargs)
            return r
        else:
            log_error('Err: request failed for URL {}, code {}'.format(url, response.code))
            r = defer.Deferred()
            return r

    agent = Agent(reactor)
    d = agent.request(
        b'GET', to_bytes(url),
        Headers({'User-Agent': ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36"]}),
        None)

    d.addCallback(check_request)
    d.addErrback(request_failed)
    return d


def run_tasks(tasks, num=10):
    def run():
        if len(tasks) > 0:
            t = tasks.pop(0)
            func = t[0]
            para = t[1:]
            d = func(*para)
            return d
        else:
            return None

    def cb(_):
        d = run()
        if d:
            d.addCallback(cb)
        return d

    dl = []
    while num > 0:
        d = run()
        if d:
            d.addCallback(cb)
            dl.append(d)
        else:
            break

        num -= 1

    defer_list = defer.DeferredList(dl)
    return defer_list
