import time
from urllib.request import urlopen
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CrawlerLimit:
    def __init__(self, time_interval=2, request_each_hour=450):
        self.last_request_time = time.time()
        self.seconds = time_interval
        self.request_count = 0
        self.request_each_hour = request_each_hour

    def get_request(self, url):
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
        return urlopen(url).read()


limit = CrawlerLimit()
