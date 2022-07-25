from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re

def download_domains(domain_name=None):
    url = 'https://trachel-srv.cs.haifa.ac.il/rachel/ppi/themes/domains.html'
    if domain_name is not None:
        url += '#name={domain_name}'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html_page = urlopen(req).read()
    soup = BeautifulSoup(html_page, 'html.parser')
    return soup

all_domains = download_domains()
domains = []
for link in all_domains.findAll('a', attrs={'href': re.compile("^domains.html#name")}):
    links.append(link.get('text'))

for domain in domains:
