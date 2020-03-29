import os
import glob
import requests
from backend.functionLib.function_lib import load_json_file, read_lines


class SSproxy:
    def __init__(self):
        self.proxy_config_folder = '/home/wangke/.ss/config'
        self.ss_local_path = '/data/wangke/ss/bin/ss-local'
        self.obfs_plugin_local_path = '/data/wangke/ss-obfs/ss/bin/obfs-local'
        self.pid_file_folder = '/home/wangke/.ss/pids'
        self.stop_ss()
        self.config_files = glob.glob(os.path.join(self.proxy_config_folder, '*.json'))
        self.proxies = []
        self.current_proxy_id = None
        os.makedirs(self.pid_file_folder, exist_ok=True)
        self.start_ss()
        print('got %d proxies: %s' % (len(self.proxies), str(self.proxies)))

    @staticmethod
    def test_proxy(proxy, test_urls=('https://ifconfig.me/ip', 'http://ifconfig.me/ip')):
        headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36"}
        for url in test_urls:
            try:
                requests.get(url, headers=headers, proxies=proxy)
            except Exception as e:
                print('proxy %s failed test in %s' % (str(proxy), url))
                print(e)
                return False
        return True

    def start_ss(self, include_none_proxy=True):
        proxies = []
        for i, config_file in enumerate(self.config_files):
            config = load_json_file(config_file)
            pid_file = os.path.join(self.pid_file_folder, '%d.pid' % i)
            cmd = "%s -c %s -f %s" % (self.ss_local_path, config_file, pid_file)
            if "obfs" in config:
                cmd += ' --plugin %s --plugin-opts "obfs=%s"' % (self.obfs_plugin_local_path, config["obfs"])
            os.system(cmd)
            port = config.get("local_port", 1080)
            local_address = config.get("local_address", "127.0.0.1")
            socks_proxy = 'socks5h://%s:%d' % (local_address, port)
            proxy = {'http': socks_proxy, "https": socks_proxy}
            if self.test_proxy(proxy):
                proxies.append(proxy)
        if include_none_proxy:
            proxies.append({'http': None, 'https': None})
        self.proxies = proxies
        self.current_proxy_id = 0

    def stop_ss(self):
        for pid_file in glob.glob(os.path.join(self.pid_file_folder, '*.pid')):
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            os.system('kill %d' % pid)

    def next_proxy(self):
        self.current_proxy_id = (self.current_proxy_id + 1) % (len(self.proxies))
        return self.proxies[self.current_proxy_id]


ssProxy = SSproxy()
