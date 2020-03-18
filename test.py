import sys, os, json, argparse
import time
import pytz
import demjson
import pandas as pd
import requests
import random, string

from twisted.internet import reactor, defer, threads
from twisted.web.client import Agent, readBody
from twisted.web.http_headers import Headers
from twisted.enterprise import adbapi
from datetime import datetime

from email.utils import parsedate_tz, mktime_tz
from dateutil import parser as date_parser

from utils import QLogger, log_info, log_error, to_bytes, to_str, http_request, run_tasks, to_timestamp
from pprint import pprint

log_path = '/data/log/ice/md_downloader.log'
#download historacal data every hour
interval_historcial = 60 * 60
interval = 60
counter = -1
db_folder = '/data/md/ice'
db_postfix = '.db'

contract_list = [   (4331, 7979),
                    (4335, 7984),
                    (236, 377),
                    (16439, 18774),
                    (15572, 17917),
                    (4337, 7982),
                ]


def get_db_file():
    return os.path.join(db_folder, datetime.now().strftime('%Y%m') + db_postfix)

def to_utctimestamp(s):
    # convert time string to utc timestamp
    if not s:
        return None
    tz_time = parsedate_tz(s)
    return  mktime_tz(tz_time) if tz_time else date_parser.parse(s).timestamp()


def download_json(url):
    def func():
        res = requests.get(url)
        res_json = demjson.decode(res.text)
        return res_json
    d = threads.deferToThread(func)
    return d

def download_contracts(url, prodID, hubID):
    d = download_json(url)
    d.addCallback(lambda x: (prodID, hubID, x))
    return d

def download_price_data(url, callback=None, *args, **kwargs):
    def cb(data):
        marketId = data['marketId']
        prices = []
        for time_str, price in data['bars']:
            prices.append((to_utctimestamp(time_str), price))

        return marketId, prices

    log_info(url)
    d = download_json(url)
    d.addCallback(cb)
    if callback:
        d.addCallback(callback, *args, **kwargs)
    return d

class DBMgr(object):
    def __init__(self):
        self.db_file = None
        self.connection = None
        self.marketIdData = None

    def load(self):
        db_file = get_db_file()
        if self.db_file != db_file:
            self.db_file = db_file
            if self.connection:
                self.connection.disconnect()

            self.connection = adbapi.ConnectionPool("sqlite3", db_file, check_same_thread=False)

            d0 = self.connection.runOperation('''create table if not exists Contracts  (
                                        productId integer, 
                                        hubId integer, 
                                        marketId integer, 
                                        lastTime integer,
                                        endDate integer,
                                        lastPrice real,
                                        marketStrip text,
                                        volume integer,
                                        change real,
                                        primary key (productId, hubId, marketId, lastTime)
                                        );'''
                                              )

            d1 = self.connection.runOperation('''create table  if not exists IntradayData (
                                            marketId integer, 
                                            datetime integer,
                                            price real,
                                            primary key (marketId, datetime)
                                            );'''
                                              )

            d2 = self.connection.runOperation('''create table  if not exists HistoricalData (
                                            marketId integer, 
                                            datetime integer,
                                            price real,
                                            primary key (marketId, datetime)
                                            );'''
                                              )

            dl = defer.DeferredList([d0, d1, d2])

            def chk(result):
                for(success, r) in result:
                    if not success:
                        log_error('create db table failed {}'.format(r))

                return self.connection
            dl.addCallback(chk)
            return dl

        else:
            return self.connection

    def persist_marketId(self, data):
        log_info('persist_marketId')
        if self.marketIdData == data:
            return
        else:
            self.marketIdData = data

        def cb(conn):
            records = []
            for entry in data.values():
                records.append('''("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")'''.format(
                    entry['productId'],
                    entry['hubId'],
                    entry['marketId'],
                    entry['lastTime'],
                    entry['endDate'],
                    entry['lastPrice'],
                    entry['marketStrip'],
                    entry['volume'],
                    entry['change'],
                ))
            v = ','.join(records)
            cmd = 'INSERT OR IGNORE INTO Contracts(productId, hubId, marketId, lastTime, endDate, lastPrice, marketStrip, volume, change) values {}'.format(v)
            d = conn.runOperation(cmd)

            def eb(e):
                log_error('Persist data failed {}'.format(e))
                log_error(cmd)
            d.addErrback(eb)
            return d

        d = defer.maybeDeferred(self.load)
        d.addCallback(cb)
        return d


    def persist_intraday_data(self, data):
        #log_info(data)
        marketId, price_data = data
        if not price_data:
            return
        #log_info('persist_intraday_data for marketId: {}'.format(marketId))
        def cb(conn):
            records = []
            for timestamp, price in price_data:
                records.append('''("{}", "{}", "{}")'''.format( marketId, timestamp, price ))
            v = ','.join(records)
            cmd = 'INSERT OR IGNORE INTO IntradayData(marketId, datetime, price) values {}'.format(v)
            d = conn.runOperation(cmd)
            def eb(e):
                log_error('Persist data failed {}'.format(e))
                log_error(cmd)
            d.addErrback(eb)
            return d

        d = defer.maybeDeferred(self.load)
        d.addCallback(cb)
        return d

    def persist_historical_data(self, data):
        #log_info(data)
        marketId, price_data = data
        if not price_data:
            return

        #log_info('persist_historical_data for marketId: {}'.format(marketId))

        def cb(conn):
            records = []
            for timestamp, price in price_data:
                records.append('''("{}", "{}", "{}")'''.format(
                    marketId,
                    timestamp,
                    price
                ))
            v = ','.join(records)
            cmd = 'INSERT OR IGNORE INTO HistoricalData(marketId, datetime, price) values {}'.format(v)
            d = conn.runOperation(cmd)
            def eb(e):
                log_error('Persist data failed {}'.format(e))
                log_error(cmd)
            d.addErrback(eb)
            return d

        d = defer.maybeDeferred(self.load)
        d.addCallback(cb)
        return d


def get_marketId():
    log_info('get_marketId')
    def chk(result):
        entries = {}
        for(success, r) in result:
            if not success:
                log_error('create db table failed {}'.format(r))
            else:
                prodID, hubID, d = r
                log_info('prodID {}, hubID {}'.format(prodID, hubID))
                for i in d:
                    i['productId'] = prodID
                    i['hubId'] = hubID
                    i['marketId'] = int(i['marketId'])
                    i['change'] = int(i['change'])
                    i['endDate'] = to_utctimestamp(i['endDate'])
                    i['lastTime'] = to_utctimestamp(i['lastTime'])
                    entries[i['marketId']] = i
        log_info('num of marketID downloaded: {}'.format(len(entries)))
        return entries

    dl = []
    for prodID, hubID in contract_list:
        url = f'https://www.theice.com/marketdata/DelayedMarkets.shtml?getContractsAsJson=&productId={prodID}&hubId={hubID}'
        log_info(url)
        d = download_contracts(url, prodID, hubID)
        dl.append(d)

    dlist = defer.DeferredList(dl)
    dlist.addCallback(chk)
    return dlist


def get_urls(marketId_entries):
    log_info('get_urls')
    urls = []
    for entry in marketId_entries.values():
        #pprint(entry)
        marketId = entry['marketId']
        intraday_url = f'https://www.theice.com/marketdata/DelayedMarkets.shtml?getIntradayChartDataAsJson=&marketId={marketId}'
        historical_url = f'https://www.theice.com/marketdata/DelayedMarkets.shtml?getHistoricalChartDataAsJson=&marketId={marketId}&historicalSpan=3'
        urls.append((intraday_url, historical_url))

    return marketId_entries, urls


dbMgr = DBMgr()
def get_prices(data):
    marketId_entries, urls = data
    dbMgr.persist_marketId(marketId_entries)
    concurrent_tasks = 10
    tasks = []

    global counter
    counter += 1
    if counter % interval_historcial == 0:
        counter = 0

    def persist_intraday(data):
        d = defer.maybeDeferred(dbMgr.persist_intraday_data, data)
        return d

    def persist_historical(data):
        d = defer.maybeDeferred(dbMgr.persist_historical_data, data)
        return d

    for intraday_url, historical_url in urls:
        tasks.append((download_price_data, intraday_url, persist_intraday))
        if counter == 0:
            tasks.append((download_price_data, historical_url, persist_historical))

    defer_list = run_tasks(tasks, concurrent_tasks)
    return defer_list

def run():
    def done(_):
        log_info('Job done at {}'.format(datetime.now().strftime("%Y%m%d %H:%M:%S")))

    log_info('Job start at {}'.format(datetime.now().strftime("%Y%m%d %H:%M:%S")))
    d = get_marketId()
    d.addCallback(get_urls)
    d.addCallback(get_prices)
    d.addBoth(done)

    reactor.callLater(interval, run)


def main(log_path_, interval_, db_folder_):
    global interval, db_folder, log_path
    interval = interval_ if interval_ else interval
    db_folder = db_folder_ if db_folder_ else db_folder
    log_path = log_path_ if log_path_ else log_path

    if not os.path.isdir(db_folder):
        log_info('sqlite sina db file does not exist, will create new one.')
        os.makedirs(db_folder)

    if not os.path.isdir(os.path.dirname(log_path)):
        log_info('sqlite sina log dir does not exist, will create new one.')
        os.makedirs(os.path.dirname(log_path))

    QLogger.startLogging(log_path)
    run()

    reactor.run()
