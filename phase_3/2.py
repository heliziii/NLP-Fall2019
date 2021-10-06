import scrapy
from scrapy.crawler import CrawlerProcess

class SemanticScholarSpider(scrapy.Spider):
    name = "SemanticScholar"
    start_urls = ["https://www.semanticscholar.org/paper/Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/998039a4876edc440e0cabb0bc42239b0eb29644",
        "https://www.semanticscholar.org/paper/Sublinear-Algorithms-for-(%CE%94%2B-1)-Vertex-Coloring-Assadi-Chen/eb4e84b8a65a21efa904b6c30ed9555278077dd3",
        "https://www.semanticscholar.org/paper/Processing-Data-Where-It-Makes-Sense%3A-Enabling-Mutlu-Ghose/4f17bd15a6f86730ac2207167ccf36ec9e6c2391"]
    current_numbers = 3
    maximum = 1000
    crawl_ids = set()
    def parse(self,response):
        authors = []
        id = response.request.url.split('/')[-1]
        self.crawl_ids.add(id)
        title = response.css('#paper-header h1 ::text').extract_first()
        print(id,title,self.current_numbers)
        abstract = response.xpath("//meta[@name='description']/@content")[0].extract()
        date = response.css('.paper-meta li:nth-child(2) span:nth-child(2) ::text').extract_first()
        for author in response.css('.author-list__author-name'):
            authors.append(author.css('span span::text').extract_first())
        references = response.css('#references .citation__title')
        ref_ids = []
        ref_links = []
        for ref in references:
            ref_id = ref.css('::attr(data-heap-paper-id)').extract_first()
            if len(ref_id) > 0:
                ref_ids.append(ref_id)
            if ref.css('a::attr(href)').extract_first() is None:
                continue
            ref_links.append(("https://www.semanticscholar.org" + ref.css('a::attr(href)').extract_first(),ref_id))
        if self.current_numbers < 1000:
            if len(self.crawl_ids) < self.maximum:
                for ref_url,ref_id in ref_links[0:5]:
                    if ref_id not in self.crawl_ids:
                        self.current_numbers += 1
                        yield scrapy.Request(ref_url, callback=self.parse)
        yield {
            'id': id,
            'title': title,
            'abstract': abstract,
            'date': date,
            'authors': authors,
            'references': ref_ids[0:10]
        }








proc = CrawlerProcess(settings={'DOWNLOAD_DELAY': 0.4, 'FEED_FORMAT': 'json', 'FEED_URI': 'result.json'})
spider = SemanticScholarSpider
proc.crawl(spider)
proc.start()
