from lxml import etree
import re
from backend.crawler.limit import Limit
from bs4 import BeautifulSoup


def is_ip_banned(r):
    if r.status_code != 403:
        return False
    html = r.content.decode('utf8')
    return 'window.location.href="https://sec.douban.com/' in html


def recode(s):
    return s.encode('utf-8').decode('utf-8')


class DoubanCrawler(object):
    def __init__(self):
        self.pre_url = 'https://movie.douban.com'

    def parse_collect(self, url, html=None):
        # s = get_etree(url)
        if html is None:
            html = Limit.get_request(url, use_proxies=True)
        s = etree.HTML(html)
        info_list = s.xpath('//div[@class="item"]/div[@class="info"]/ul')
        res = []
        for info in info_list:
            movie_href = info.xpath('./li[@class="title"]/a/@href')[0]
            movie_id = re.search(r'\d+', recode(movie_href)).group()
            spans = info.xpath('./li/span')
            date = recode(info.xpath('./li/span[@class="date"]/text()')[0])
            rate = re.search(r'\d', spans[0].get('class'))
            if rate is not None:
                rate = int(rate.group())
            else:
                rate = 0
            comment = (info.xpath('./li/span[@class="comment"]/text()') or [''])[0]
            res.append({'movie_id': movie_id, 'rate': rate, 'comment': comment, 'date': date})
        next_page_url = (s.xpath('//span[@class="next"]/a/@href') or [None])[0]
        if next_page_url is not None:
            next_page_url = self.pre_url + next_page_url
        return res, next_page_url, html

    def get_collect(self, uid):
        url = '%s/people/%s/collect' % (self.pre_url, uid)
        collects = []
        htmls = []
        while url is not None:
            res, next_url, html = self.parse_collect(url)
            collects.extend(res)
            htmls.append({'url': url, 'content': html})
            url = next_url
        return collects, htmls

    def parse_user_review_page(self, url, html=None):
        if html is None:
            html = Limit.get_request(url, use_proxies=True)
        s = etree.HTML(html)
        next_page_url = (s.xpath('//span[@class="next"]/a/@href') or [None])[0]
        review_urls = s.xpath('//li[@class="nlst"]/h3/a/@href') or []
        return review_urls, next_page_url, html

    def get_user_review_list(self, uid, htmls=None,review_num_unmatch_warning=True):
        url = 'https://movie.douban.com/people/%s/reviews' % uid
        review_urls = []
        if htmls is None:
            htmls = []
            now_url = url
            while now_url is not None:
                try:
                    res, next_url, html = self.parse_user_review_page(now_url)
                except Exception as e:
                    print('Error in url %s: %s' % (now_url, str(e)))
                    break
                review_urls.extend(res)
                htmls.append({'url': now_url, 'content': html})
                if next_url is None:
                    break
                now_url = url + next_url
        else:
            for html in [x["content"] for x in htmls]:
                res, _, _ = self.parse_user_review_page(None, html)
                review_urls.extend(res)
        s = etree.HTML(htmls[0]["content"])
        title = recode((s.xpath('//div[@id="db-usr-profile"]/div[@class="info"]/h1/text()') or [''])[0])
        if title == '':
            print('Error in page %s, check this page and maybe your cache html' % url)
        else:
            review_num = int(title.split('(')[-1].split(')')[0])
            if review_num != len(review_urls) and review_num_unmatch_warning:
                print('user: %s, expected: %d, got: %d' % (uid, review_num, len(review_urls)))
        return review_urls, htmls

    def get_user_review_htmls(self, review_urls):
        htmls = []
        for url in review_urls:
            try:
                html = Limit.get_request(url, use_proxies=True, need_sleep=is_ip_banned)
            except Exception as e:
                print('Error in url %s: %s' % (url, str(e)))
                html = ''
            htmls.append({'url': url, 'content': html})
        return htmls

    def get_user_reviews(self, htmls):
        reviews = []
        for url, html in [(x["url"], x["content"]) for x in htmls]:
            if html == '':
                reviews.append({})
                continue
            s = etree.HTML(html)
            title = recode((s.xpath('//div[@class="article"]/h1/span/text()') or [''])[0])
            movie_href = recode(s.xpath('//header[@class="main-hd"]/a/@href')[1])
            movie_id = re.search(r'\d+', movie_href).group()
            try:
                rate = int(s.xpath('//span[@class="main-title-hide"]/text()')[0])
            except IndexError:
                rate = 0
                print('unrated review: %s' % url)
            date = recode(s.xpath('//span[@class="main-meta"]/text()')[0])
            content = '\n'.join(
                map(recode, s.xpath('//div[@id="link-report"]/div[@data-author]/descendant-or-self::*/text()')))
            votes = [recode(x) for x in s.xpath('//div[@class="main-panel-useful"]/descendant::*/text()')]
            if len(votes) == 0:
                soup = BeautifulSoup(html)
                votes = [x.text.strip() for x in soup.find(attrs={'class': "main-panel-useful"}).find_all("button")]
            useful = re.search(r'\d+', votes[0])
            if useful is None:
                useful = 0
            else:
                useful = useful.group()
            useless = re.search(r'\d+', votes[1])
            if useless is None:
                useless = 0
            else:
                useless = useless.group()
            reviews.append(
                {'title': title, 'movie': movie_id, 'rate': rate, 'date': date, 'content': content, 'useful': useful,
                 'useless': useless})
        return reviews


douban_crawler = DoubanCrawler()
