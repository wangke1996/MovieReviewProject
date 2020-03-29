import time
from urllib.request import urlopen, Request
import ssl
import string
from fake_useragent import UserAgent
from backend.functionLib.function_lib import logging_with_time
import requests
import random
from .ss_proxys import ssProxy

ua = UserAgent()
ssl._create_default_https_context = ssl._create_unverified_context


# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}

def fake_cookie():
    cookies = [
        r'_vwo_uuid_v2=DBF8731551621D71C6FA0C43E82710E79|11d92d9b73e1192c74f15839464e4633; douban-fav-remind=1; bid=JpLuSx5PwX4; __yadk_uid=TdG8uCVWOGo4lWU2ruAUPLECW6SG1ybh; trc_cookie_storage=taboola%2520global%253Auser-id%3D274659d2-521d-4532-a0fb-ac2d685e46aa-tuct1902a87; ll="118340"; __gads=ID=d2952a747ce9a64d:T=1581319910:S=ALNI_Mauyj_wbfpbNW8vHD0sgnhZDFJIbg; ct=y; __utmc=30149280; push_doumail_num=0; push_noty_num=0; __utmz=30149280.1581753960.6.4.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1581785026%2C%22https%3A%2F%2Faccounts.douban.com%2Fpassport%2Flogin%3Fsource%3Dmain%22%5D; _pk_ses.100001.8cb4=*; ap_v=0,6.0; __utma=30149280.1953762817.1581697490.1581753960.1581785028.7; ck=QXCe; __utmv=30149280.20997; __utmb=30149280.28.10.1581785028; _pk_id.100001.8cb4=65c0f41717601bde.1524134928.22.1581787551.1581753958.; dbcl2="209508158:RiX1EySEzG0"',
    ]
    return {'cookie': "bid=%s; " % "".join(random.sample(string.ascii_letters + string.digits, 11))}
    # return {'cookie': "bid=%s; " % "".join(random.sample(string.ascii_letters + string.digits, 11)) + random.choice(
    #     cookies)}


class CrawlerLimit:
    def __init__(self, time_interval=2, request_each_hour=450):
        self.last_request_time = time.time()
        self.seconds = time_interval
        self.request_count = 0
        self.request_each_hour = request_each_hour
        self.init_sleep_time = 10 * 60
        self.sleep_time_for_error_status = self.init_sleep_time
        self.max_sleep_time = 60 * 60 * 2
        self.retry_status = 403

    def set_retry_status(self, retry_status='all', init_sleep_time=600):
        self.retry_status = retry_status
        self.init_sleep_time = init_sleep_time
        self.sleep_time_for_error_status = init_sleep_time

    def get_request(self, url, use_cookie=False, use_proxies=False, need_sleep=None):
        now_time = time.time()
        # if self.request_count >= self.request_each_hour:
        #     print('sleep for one hour...')
        #     time.sleep(3600)
        #     self.request_count = 0
        sleep_time = self.last_request_time + 2 - now_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_request_time = now_time
        self.request_count = self.request_count + 1
        while True:
            headers = {'User-Agent': ua.random}
            proxy = {'http': None, 'https': None}
            if use_proxies:
                proxy = ssProxy.next_proxy()
            if use_cookie:
                r = requests.get(url, cookies=fake_cookie(), headers=headers, proxies=proxy)
            else:
                r = requests.get(url, headers=headers, proxies=proxy)
            status = r.status_code
            if (self.retry_status == 'all' and status != 200) or status == self.retry_status:
                if need_sleep is not None and not need_sleep(r):
                    break
                logging_with_time(
                    '%d for %s, will sleep for %d seconds' % (status, url, self.sleep_time_for_error_status))
                time.sleep(self.sleep_time_for_error_status)
                self.sleep_time_for_error_status = min(self.sleep_time_for_error_status * 2, self.max_sleep_time)
                r.close()
            else:
                self.sleep_time_for_error_status = self.init_sleep_time
                break
        content = r.content.decode('utf-8')
        r.close()
        r.raise_for_status()
        return content


Limit = CrawlerLimit()
