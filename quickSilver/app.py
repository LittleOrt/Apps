from flask import Flask, render_template, request, jsonify, send_file
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import pandas as pd
import re
from datetime import datetime
import io
import json
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
import unicodedata

app = Flask(__name__)

# Download NLTK data on first run
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data"""
    required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    for data in required_data:
        try:
            nltk.data.find(f'tokenizers/{data}')
        except LookupError:
            try:
                nltk.download(data, quiet=True)
            except:
                pass

download_nltk_data()

class DataCleaner:
    """Advanced data cleaning and standardization"""
    
    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\r\t')
        
        return text.strip()
    
    @staticmethod
    def clean_url(url, base_url):
        """Clean and validate URL"""
        if not url:
            return ""
        
        url = str(url).strip()
        
        # Handle relative URLs
        if not url.startswith(('http://', 'https://', '//')):
            url = urljoin(base_url, url)
        
        # Handle protocol-relative URLs
        if url.startswith('//'):
            url = 'https:' + url
        
        return url
    
    @staticmethod
    def clean_numeric(value):
        """Extract and clean numeric values"""
        if not value:
            return ""
        
        # Extract numbers from string
        numbers = re.findall(r'-?\d+\.?\d*', str(value))
        return numbers[0] if numbers else ""
    
    @staticmethod
    def standardize_phone(phone):
        """Standardize phone number format"""
        if not phone:
            return ""
        
        # Remove non-digit characters except + at start
        digits = re.sub(r'[^\d+]', '', str(phone))
        
        # Format: +1-XXX-XXX-XXXX or XXX-XXX-XXXX
        if len(digits) >= 10:
            return digits
        
        return phone
    
    @staticmethod
    def validate_email(email):
        """Validate email format"""
        if not email:
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, str(email)))

class IntelligentElementMatcher:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.lemmatizer = None
        
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'price': r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars?)\b',
            'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
            'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
            'social_media': r'(?:@[A-Za-z0-9_]+|(?:facebook|twitter|instagram|linkedin)\.com/[A-Za-z0-9_.-]+)'
        }
        
        self.element_keywords = {
            'title': ['title', 'heading', 'header', 'h1', 'h2', 'h3', 'headline', 'caption', 'name'],
            'description': ['description', 'desc', 'summary', 'content', 'text', 'paragraph', 'detail', 'about', 'info'],
            'image': ['image', 'img', 'picture', 'photo', 'thumbnail', 'graphic', 'visual', 'media'],
            'link': ['link', 'url', 'href', 'anchor', 'hyperlink', 'reference'],
            'table': ['table', 'grid', 'data', 'list', 'spreadsheet', 'chart'],
            'contact': ['email', 'phone', 'contact', 'address', 'tel', 'telephone', 'mail'],
            'price': ['price', 'cost', 'amount', 'value', '$', 'fee', 'rate', 'charge', 'payment'],
            'date': ['date', 'time', 'published', 'updated', 'timestamp', 'when', 'schedule'],
            'product': ['product', 'item', 'goods', 'merchandise', 'article'],
            'author': ['author', 'writer', 'by', 'creator', 'publisher'],
            'category': ['category', 'tag', 'topic', 'section', 'type', 'class'],
            'social': ['social', 'twitter', 'facebook', 'instagram', 'linkedin', 'share'],
            'video': ['video', 'youtube', 'vimeo', 'media', 'embed'],
            'form': ['form', 'input', 'search', 'subscribe', 'newsletter'],
            'meta': ['meta', 'metadata', 'seo', 'keywords', 'tags']
        }
    
    def preprocess_query(self, query):
        """Preprocess user query using NLP"""
        if not query:
            return []
        
        try:
            tokens = word_tokenize(query.lower())
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token.isalnum() and token not in self.stop_words]
            else:
                tokens = [token for token in tokens if token.isalnum()]
            return tokens
        except:
            return query.lower().split()
    
    def match_intent(self, user_query):
        """Match user query to scraping intent using NLP"""
        if not user_query:
            return ['all']
        
        tokens = self.preprocess_query(user_query)
        matched_types = set()
        
        scores = {}
        for element_type, keywords in self.element_keywords.items():
            score = 0
            for token in tokens:
                for keyword in keywords:
                    if token in keyword or keyword in token:
                        score += 1
            if score > 0:
                scores[element_type] = score
        
        if scores:
            max_score = max(scores.values())
            matched_types = [k for k, v in scores.items() if v >= max_score * 0.7]
        
        return list(matched_types) if matched_types else ['all']
    
    def extract_pattern(self, text, pattern_type):
        """Extract specific patterns from text"""
        if pattern_type in self.patterns:
            pattern = self.patterns[pattern_type]
            return re.findall(pattern, text, re.IGNORECASE)
        return []

class WebScraper:
    def __init__(self, url, timeout=20):
        self.url = url
        self.timeout = timeout
        self.soup = None
        self.base_url = None
        self.errors = []
        self.cleaner = DataCleaner()
        
    def validate_url(self):
        """Validate URL format"""
        try:
            result = urlparse(self.url)
            if not all([result.scheme, result.netloc]):
                return False, "Invalid URL format. Please include http:// or https://"
            if result.scheme not in ['http', 'https']:
                return False, "URL must start with http:// or https://"
            return True, ""
        except Exception as e:
            return False, f"URL validation error: {str(e)}"
    
    def fetch_page(self):
        """Fetch and parse the webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(self.url, headers=headers, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            self.soup = BeautifulSoup(response.content, 'html.parser')
            self.base_url = f"{urlparse(self.url).scheme}://{urlparse(self.url).netloc}"
            
            return True, "Page fetched successfully"
            
        except requests.exceptions.Timeout:
            return False, "⏱️ Connection timeout. The website took too long to respond."
        except requests.exceptions.ConnectionError:
            return False, "🌐 Connection failed. Check your internet connection or verify the URL."
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else 0
            if status_code == 403:
                return False, "🚫 Access forbidden (403). The website blocks automated access."
            elif status_code == 404:
                return False, "❌ Page not found (404). Verify the URL is correct."
            elif status_code == 429:
                return False, "⏸️ Too many requests (429). Wait and try again."
            else:
                return False, f"⚠️ HTTP Error {status_code}"
        except Exception as e:
            return False, f"❗ Unexpected error: {str(e)}"
    
    def get_available_content(self):
        """Analyze page and return available content types with counts"""
        content_analysis = {}
        
        # Titles/Headings
        headings = self.soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if headings:
            content_analysis['titles'] = {
                'count': len(headings),
                'preview': self.cleaner.clean_text(headings[0].get_text())[:100] if headings else ""
            }
        
        # Paragraphs/Descriptions
        paragraphs = self.soup.find_all('p')
        if paragraphs:
            content_analysis['descriptions'] = {
                'count': len(paragraphs),
                'preview': self.cleaner.clean_text(paragraphs[0].get_text())[:100] if paragraphs else ""
            }
        
        # Images
        images = self.soup.find_all('img')
        if images:
            content_analysis['images'] = {
                'count': len(images),
                'preview': f"{images[0].get('src', 'No src')[:50]}..." if images else ""
            }
        
        # Links
        links = self.soup.find_all('a', href=True)
        if links:
            content_analysis['links'] = {
                'count': len(links),
                'preview': links[0].get_text(strip=True)[:50] if links else ""
            }
        
        # Tables
        tables = self.soup.find_all('table')
        if tables:
            content_analysis['tables'] = {
                'count': len(tables),
                'preview': f"Table with {len(tables[0].find_all('tr'))} rows" if tables else ""
            }
        
        # Contact info (emails & phones)
        page_text = self.soup.get_text()
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', page_text)
        phones = re.findall(r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b', page_text)
        
        if emails or phones:
            content_analysis['contact'] = {
                'count': len(set(emails)) + len(set(phones)),
                'preview': f"{len(set(emails))} emails, {len(set(phones))} phones"
            }
        
        # Prices
        prices = re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', page_text)
        if prices:
            content_analysis['prices'] = {
                'count': len(set(prices)),
                'preview': prices[0] if prices else ""
            }
        
        # Videos
        videos = self.soup.find_all(['video', 'iframe'])
        if videos:
            content_analysis['videos'] = {
                'count': len(videos),
                'preview': "Video content detected"
            }
        
        # Forms
        forms = self.soup.find_all('form')
        if forms:
            content_analysis['forms'] = {
                'count': len(forms),
                'preview': f"{len(forms)} form(s) detected"
            }
        
        # Metadata
        meta_tags = self.soup.find_all('meta')
        if meta_tags:
            content_analysis['meta'] = {
                'count': len(meta_tags),
                'preview': f"{len(meta_tags)} meta tags"
            }
        
        return content_analysis
    
    def scrape_full(self, selected_features=None):
        """Complete scrape with no omissions - gets ALL data"""
        results = {
            'metadata': {
                'url': self.url,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'title': '',
                'description': ''
            },
            'data': {}
        }
        
        try:
            # Get page metadata
            title_tag = self.soup.find('title')
            results['metadata']['title'] = self.cleaner.clean_text(title_tag.string) if title_tag else 'No title'
            
            meta_desc = self.soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                results['metadata']['description'] = self.cleaner.clean_text(meta_desc['content'])
            
            # If no features selected, scrape everything
            if not selected_features:
                selected_features = ['all']
            
            # Extract ALL data based on selected features (no limits!)
            if 'all' in selected_features or 'titles' in selected_features:
                results['data']['titles'] = self._extract_all_titles()
            
            if 'all' in selected_features or 'descriptions' in selected_features:
                results['data']['descriptions'] = self._extract_all_descriptions()
            
            if 'all' in selected_features or 'images' in selected_features:
                results['data']['images'] = self._extract_all_images()
            
            if 'all' in selected_features or 'links' in selected_features:
                results['data']['links'] = self._extract_all_links()
            
            if 'all' in selected_features or 'tables' in selected_features:
                results['data']['tables'] = self._extract_all_tables()
            
            if 'all' in selected_features or 'contact' in selected_features:
                results['data']['contact_info'] = self._extract_all_contact_info()
            
            if 'all' in selected_features or 'prices' in selected_features:
                results['data']['prices'] = self._extract_all_prices()
            
            if 'all' in selected_features or 'videos' in selected_features:
                results['data']['videos'] = self._extract_all_videos()
            
            if 'all' in selected_features or 'forms' in selected_features:
                results['data']['forms'] = self._extract_all_forms()
            
            if 'all' in selected_features or 'meta' in selected_features:
                results['data']['meta'] = self._extract_all_meta()
            
            return results
            
        except Exception as e:
            self.errors.append(f"Scraping error: {str(e)}")
            return results
    
    def _extract_all_titles(self):
        """Extract ALL headings - no limits"""
        titles = []
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for idx, element in enumerate(self.soup.find_all(tag)):
                text = self.cleaner.clean_text(element.get_text())
                if text:
                    titles.append({
                        'level': tag.upper(),
                        'text': text,
                        'position': idx + 1,
                        'id': element.get('id', ''),
                        'class': ' '.join(element.get('class', []))
                    })
        return titles
    
    def _extract_all_descriptions(self):
        """Extract ALL paragraphs - complete"""
        descriptions = []
        
        # Meta description
        meta_desc = self.soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            descriptions.append({
                'type': 'Meta Description',
                'text': self.cleaner.clean_text(meta_desc['content']),
                'length': len(meta_desc['content']),
                'position': 0
            })
        
        # ALL paragraphs
        for idx, p in enumerate(self.soup.find_all('p')):
            text = self.cleaner.clean_text(p.get_text())
            if text:
                descriptions.append({
                    'type': 'Paragraph',
                    'text': text,
                    'length': len(text),
                    'position': idx + 1
                })
        
        return descriptions
    
    def _extract_all_images(self):
        """Extract ALL images - complete"""
        images = []
        
        for idx, img in enumerate(self.soup.find_all('img')):
            src = img.get('src', '') or img.get('data-src', '') or img.get('data-lazy-src', '')
            if src:
                full_url = self.cleaner.clean_url(src, self.base_url)
                images.append({
                    'position': idx + 1,
                    'url': full_url,
                    'alt': self.cleaner.clean_text(img.get('alt', '')),
                    'title': self.cleaner.clean_text(img.get('title', '')),
                    'width': img.get('width', ''),
                    'height': img.get('height', ''),
                    'loading': img.get('loading', '')
                })
        
        return images
    
    def _extract_all_links(self):
        """Extract ALL links - complete"""
        links = []
        
        for idx, a in enumerate(self.soup.find_all('a', href=True)):
            href = a['href']
            full_url = self.cleaner.clean_url(href, self.base_url)
            link_text = self.cleaner.clean_text(a.get_text())
            
            links.append({
                'position': idx + 1,
                'url': full_url,
                'text': link_text,
                'title': self.cleaner.clean_text(a.get('title', '')),
                'is_external': self.base_url not in full_url,
                'rel': ' '.join(a.get('rel', [])),
                'target': a.get('target', '')
            })
        
        return links
    
    def _extract_all_tables(self):
        """Extract ALL tables completely"""
        all_tables = []
        
        for table_idx, table in enumerate(self.soup.find_all('table')):
            headers = []
            rows_data = []
            
            # Extract headers
            header_row = table.find('thead')
            if header_row:
                headers = [self.cleaner.clean_text(th.get_text()) for th in header_row.find_all(['th', 'td'])]
            else:
                first_row = table.find('tr')
                if first_row:
                    headers = [self.cleaner.clean_text(th.get_text()) for th in first_row.find_all(['th', 'td'])]
            
            # Generate headers if none found
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    num_cols = len(first_row.find_all(['td', 'th']))
                    headers = [f'Column_{i+1}' for i in range(num_cols)]
            
            # Extract ALL data rows
            tbody = table.find('tbody')
            rows_to_process = tbody.find_all('tr') if tbody else table.find_all('tr')
            
            # Skip first row if it was used for headers
            start_idx = 1 if not table.find('thead') and headers else 0
            
            for row in rows_to_process[start_idx:]:
                cells = [self.cleaner.clean_text(td.get_text()) for td in row.find_all(['td', 'th'])]
                if cells and any(cells):
                    rows_data.append(cells)
            
            if rows_data and headers:
                all_tables.append({
                    'table_number': table_idx + 1,
                    'headers': headers,
                    'rows': rows_data,
                    'row_count': len(rows_data),
                    'column_count': len(headers)
                })
        
        return all_tables
    
    def _extract_all_contact_info(self):
        """Extract ALL contact information"""
        matcher = IntelligentElementMatcher()
        page_text = self.soup.get_text()
        
        emails = list(set(matcher.extract_pattern(page_text, 'email')))
        phones = list(set(matcher.extract_pattern(page_text, 'phone')))
        
        # Clean and validate
        emails = [e for e in emails if self.cleaner.validate_email(e)]
        phones = [self.cleaner.standardize_phone(p) for p in phones]
        
        return {
            'emails': emails,
            'phones': phones,
            'total_contacts': len(emails) + len(phones)
        }
    
    def _extract_all_prices(self):
        """Extract ALL prices with context"""
        matcher = IntelligentElementMatcher()
        prices = []
        
        # Find ALL price-related elements
        price_elements = self.soup.find_all(['span', 'div', 'p', 'strong', 'b'], 
                                            class_=re.compile(r'price|cost|amount', re.I))
        
        for elem in price_elements:
            text = self.cleaner.clean_text(elem.get_text())
            found_prices = matcher.extract_pattern(text, 'price')
            for price in found_prices:
                prices.append({
                    'price': price,
                    'context': text[:150]
                })
        
        # Also search entire page text
        page_text = self.soup.get_text()
        general_prices = matcher.extract_pattern(page_text, 'price')
        
        for price in general_prices:
            if not any(p['price'] == price for p in prices):
                prices.append({
                    'price': price,
                    'context': 'General page content'
                })
        
        return prices
    
    def _extract_all_videos(self):
        """Extract ALL videos"""
        videos = []
        
        # Video tags
        for idx, video in enumerate(self.soup.find_all('video')):
            videos.append({
                'position': idx + 1,
                'type': 'video',
                'src': self.cleaner.clean_url(video.get('src', ''), self.base_url),
                'poster': self.cleaner.clean_url(video.get('poster', ''), self.base_url),
                'width': video.get('width', ''),
                'height': video.get('height', '')
            })
        
        # iframes (YouTube, Vimeo, etc.)
        for idx, iframe in enumerate(self.soup.find_all('iframe')):
            src = iframe.get('src', '')
            if any(platform in src.lower() for platform in ['youtube', 'vimeo', 'dailymotion', 'video']):
                videos.append({
                    'position': len(videos) + 1,
                    'type': 'iframe',
                    'src': self.cleaner.clean_url(src, self.base_url),
                    'title': self.cleaner.clean_text(iframe.get('title', '')),
                    'width': iframe.get('width', ''),
                    'height': iframe.get('height', '')
                })
        
        return videos
    
    def _extract_all_forms(self):
        """Extract ALL forms with complete details including hidden/locked elements"""
        forms = []
        
        for form_idx, form in enumerate(self.soup.find_all('form')):
            form_data = {
                'form_number': form_idx + 1,
                'form_id': form.get('id', ''),
                'form_name': form.get('name', ''),
                'form_class': ' '.join(form.get('class', [])),
                'action': self.cleaner.clean_url(form.get('action', ''), self.base_url),
                'method': form.get('method', 'GET').upper(),
                'enctype': form.get('enctype', ''),
                'autocomplete': form.get('autocomplete', ''),
                'target': form.get('target', ''),
                'fields': []
            }
            
            # Extract ALL form elements recursively
            field_position = 0
            
            # Input elements (including hidden, disabled, readonly)
            for input_elem in form.find_all('input'):
                field_position += 1
                input_type = input_elem.get('type', 'text').lower()
                
                field_data = {
                    'position': field_position,
                    'element_type': 'input',
                    'input_type': input_type,
                    'name': input_elem.get('name', ''),
                    'id': input_elem.get('id', ''),
                    'value': self.cleaner.clean_text(input_elem.get('value', '')),
                    'placeholder': self.cleaner.clean_text(input_elem.get('placeholder', '')),
                    'label': self._find_label_for_element(form, input_elem),
                    'required': input_elem.has_attr('required'),
                    'disabled': input_elem.has_attr('disabled'),
                    'readonly': input_elem.has_attr('readonly'),
                    'hidden': input_type == 'hidden' or input_elem.has_attr('hidden'),
                    'checked': input_elem.has_attr('checked'),
                    'min': input_elem.get('min', ''),
                    'max': input_elem.get('max', ''),
                    'maxlength': input_elem.get('maxlength', ''),
                    'pattern': input_elem.get('pattern', ''),
                    'class': ' '.join(input_elem.get('class', [])),
                    'aria_label': input_elem.get('aria-label', ''),
                    'data_attributes': self._extract_data_attributes(input_elem)
                }
                form_data['fields'].append(field_data)
            
            # Textarea elements
            for textarea in form.find_all('textarea'):
                field_position += 1
                field_data = {
                    'position': field_position,
                    'element_type': 'textarea',
                    'input_type': 'textarea',
                    'name': textarea.get('name', ''),
                    'id': textarea.get('id', ''),
                    'value': self.cleaner.clean_text(textarea.get_text()),
                    'placeholder': self.cleaner.clean_text(textarea.get('placeholder', '')),
                    'label': self._find_label_for_element(form, textarea),
                    'required': textarea.has_attr('required'),
                    'disabled': textarea.has_attr('disabled'),
                    'readonly': textarea.has_attr('readonly'),
                    'hidden': textarea.has_attr('hidden'),
                    'rows': textarea.get('rows', ''),
                    'cols': textarea.get('cols', ''),
                    'maxlength': textarea.get('maxlength', ''),
                    'class': ' '.join(textarea.get('class', [])),
                    'aria_label': textarea.get('aria-label', ''),
                    'data_attributes': self._extract_data_attributes(textarea)
                }
                form_data['fields'].append(field_data)
            
            # Select elements (dropdowns)
            for select in form.find_all('select'):
                field_position += 1
                
                # Extract all options
                options = []
                for option in select.find_all('option'):
                    options.append({
                        'value': option.get('value', ''),
                        'text': self.cleaner.clean_text(option.get_text()),
                        'selected': option.has_attr('selected'),
                        'disabled': option.has_attr('disabled')
                    })
                
                field_data = {
                    'position': field_position,
                    'element_type': 'select',
                    'input_type': 'select',
                    'name': select.get('name', ''),
                    'id': select.get('id', ''),
                    'label': self._find_label_for_element(form, select),
                    'required': select.has_attr('required'),
                    'disabled': select.has_attr('disabled'),
                    'hidden': select.has_attr('hidden'),
                    'multiple': select.has_attr('multiple'),
                    'size': select.get('size', ''),
                    'options': options,
                    'options_count': len(options),
                    'class': ' '.join(select.get('class', [])),
                    'aria_label': select.get('aria-label', ''),
                    'data_attributes': self._extract_data_attributes(select)
                }
                form_data['fields'].append(field_data)
            
            # Button elements
            for button in form.find_all('button'):
                field_position += 1
                field_data = {
                    'position': field_position,
                    'element_type': 'button',
                    'input_type': button.get('type', 'button'),
                    'name': button.get('name', ''),
                    'id': button.get('id', ''),
                    'value': self.cleaner.clean_text(button.get('value', '')),
                    'text': self.cleaner.clean_text(button.get_text()),
                    'disabled': button.has_attr('disabled'),
                    'class': ' '.join(button.get('class', [])),
                    'aria_label': button.get('aria-label', ''),
                    'data_attributes': self._extract_data_attributes(button)
                }
                form_data['fields'].append(field_data)
            
            # Submit buttons (input type="submit")
            for submit_btn in form.find_all('input', type='submit'):
                if not any(f['id'] == submit_btn.get('id', '') and f['name'] == submit_btn.get('name', '') 
                          for f in form_data['fields']):
                    field_position += 1
                    field_data = {
                        'position': field_position,
                        'element_type': 'input',
                        'input_type': 'submit',
                        'name': submit_btn.get('name', ''),
                        'id': submit_btn.get('id', ''),
                        'value': self.cleaner.clean_text(submit_btn.get('value', 'Submit')),
                        'disabled': submit_btn.has_attr('disabled'),
                        'class': ' '.join(submit_btn.get('class', []))
                    }
                    form_data['fields'].append(field_data)
            
            form_data['total_fields'] = len(form_data['fields'])
            form_data['hidden_fields'] = sum(1 for f in form_data['fields'] if f.get('hidden', False))
            form_data['required_fields'] = sum(1 for f in form_data['fields'] if f.get('required', False))
            form_data['disabled_fields'] = sum(1 for f in form_data['fields'] if f.get('disabled', False))
            
            forms.append(form_data)
        
        return forms
    
    def _find_label_for_element(self, form, element):
        """Find associated label for form element"""
        element_id = element.get('id', '')
        element_name = element.get('name', '')
        
        # Try to find label by 'for' attribute
        if element_id:
            label = form.find('label', attrs={'for': element_id})
            if label:
                return self.cleaner.clean_text(label.get_text())
        
        # Try to find parent label
        parent_label = element.find_parent('label')
        if parent_label:
            return self.cleaner.clean_text(parent_label.get_text())
        
        # Try to find nearby label by position
        prev_sibling = element.find_previous_sibling('label')
        if prev_sibling:
            return self.cleaner.clean_text(prev_sibling.get_text())
        
        # Try aria-label
        aria_label = element.get('aria-label', '')
        if aria_label:
            return self.cleaner.clean_text(aria_label)
        
        # Use placeholder or name as fallback
        placeholder = element.get('placeholder', '')
        if placeholder:
            return self.cleaner.clean_text(placeholder)
        
        if element_name:
            return element_name.replace('_', ' ').replace('-', ' ').title()
        
        return ''
    
    def _extract_data_attributes(self, element):
        """Extract all data-* attributes from element"""
        data_attrs = {}
        for attr, value in element.attrs.items():
            if attr.startswith('data-'):
                data_attrs[attr] = self.cleaner.clean_text(str(value))
        return data_attrs
    
    def _extract_all_meta(self):
        """Extract ALL meta tags"""
        meta_tags = []
        
        for idx, meta in enumerate(self.soup.find_all('meta')):
            meta_data = {
                'position': idx + 1,
                'name': meta.get('name', ''),
                'property': meta.get('property', ''),
                'content': self.cleaner.clean_text(meta.get('content', '')),
                'http_equiv': meta.get('http-equiv', '')
            }
            meta_tags.append(meta_data)
        
        return meta_tags

def create_clean_csv_data(scrape_data):
    """Create clean, standardized CSV data ready for Excel"""
    data_frames = {}
    
    # Metadata sheet
    metadata_df = pd.DataFrame([{
        'URL': scrape_data['metadata']['url'],
        'Page_Title': scrape_data['metadata']['title'],
        'Description': scrape_data['metadata']['description'],
        'Scraped_DateTime': scrape_data['metadata']['timestamp']
    }])
    data_frames['1_Metadata'] = metadata_df
    
    # Process each data category with proper cleaning
    for category, items in scrape_data.get('data', {}).items():
        if not items:
            continue
        
        if category == 'titles' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['2_Titles'] = df
        
        elif category == 'descriptions' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['3_Descriptions'] = df
        
        elif category == 'images' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['4_Images'] = df
        
        elif category == 'links' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['5_Links'] = df
        
        elif category == 'tables' and isinstance(items, list):
            for table_data in items:
                table_num = table_data.get('table_number', 1)
                headers = table_data.get('headers', [])
                rows = table_data.get('rows', [])
                
                if rows and headers:
                    # Ensure consistent column count
                    max_cols = len(headers)
                    normalized_rows = []
                    
                    for row in rows:
                        # Pad or trim row to match header length
                        if len(row) < max_cols:
                            row = row + [''] * (max_cols - len(row))
                        elif len(row) > max_cols:
                            row = row[:max_cols]
                        normalized_rows.append(row)
                    
                    df = pd.DataFrame(normalized_rows, columns=headers)
                    df = df.fillna('')
                    data_frames[f'6_Table_{table_num}'] = df
        
        elif category == 'contact_info' and isinstance(items, dict):
            contact_rows = []
            
            for email in items.get('emails', []):
                contact_rows.append({
                    'Type': 'Email',
                    'Value': email,
                    'Validated': 'Yes'
                })
            
            for phone in items.get('phones', []):
                contact_rows.append({
                    'Type': 'Phone',
                    'Value': phone,
                    'Validated': 'Yes'
                })
            
            if contact_rows:
                df = pd.DataFrame(contact_rows)
                data_frames['7_Contact_Info'] = df
        
        elif category == 'prices' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['8_Prices'] = df
        
        elif category == 'videos' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['9_Videos'] = df
        
        elif category == 'forms' and isinstance(items, list) and items:
            # Create comprehensive form sheets
            for form in items:
                form_num = form.get('form_number', 1)
                
                # Sheet 1: Form Overview
                form_overview = {
                    'Form_Number': form_num,
                    'Form_ID': form.get('form_id', ''),
                    'Form_Name': form.get('form_name', ''),
                    'Form_Class': form.get('form_class', ''),
                    'Action_URL': form.get('action', ''),
                    'Method': form.get('method', ''),
                    'Encoding_Type': form.get('enctype', ''),
                    'Autocomplete': form.get('autocomplete', ''),
                    'Target': form.get('target', ''),
                    'Total_Fields': form.get('total_fields', 0),
                    'Hidden_Fields': form.get('hidden_fields', 0),
                    'Required_Fields': form.get('required_fields', 0),
                    'Disabled_Fields': form.get('disabled_fields', 0)
                }
                overview_df = pd.DataFrame([form_overview])
                data_frames[f'10_Form_{form_num}_Overview'] = overview_df
                
                # Sheet 2: Form Fields Details
                fields = form.get('fields', [])
                if fields:
                    fields_data = []
                    for field in fields:
                        field_row = {
                            'Position': field.get('position', ''),
                            'Element_Type': field.get('element_type', ''),
                            'Input_Type': field.get('input_type', ''),
                            'Name': field.get('name', ''),
                            'ID': field.get('id', ''),
                            'Label': field.get('label', ''),
                            'Value': field.get('value', ''),
                            'Placeholder': field.get('placeholder', ''),
                            'Required': 'Yes' if field.get('required', False) else 'No',
                            'Disabled': 'Yes' if field.get('disabled', False) else 'No',
                            'Readonly': 'Yes' if field.get('readonly', False) else 'No',
                            'Hidden': 'Yes' if field.get('hidden', False) else 'No',
                            'Checked': 'Yes' if field.get('checked', False) else 'No',
                            'Min': field.get('min', ''),
                            'Max': field.get('max', ''),
                            'MaxLength': field.get('maxlength', ''),
                            'Pattern': field.get('pattern', ''),
                            'Rows': field.get('rows', ''),
                            'Cols': field.get('cols', ''),
                            'Multiple': 'Yes' if field.get('multiple', False) else 'No',
                            'Size': field.get('size', ''),
                            'Options_Count': field.get('options_count', ''),
                            'Class': field.get('class', ''),
                            'ARIA_Label': field.get('aria_label', ''),
                            'Text': field.get('text', ''),
                            'Data_Attributes': json.dumps(field.get('data_attributes', {}))
                        }
                        fields_data.append(field_row)
                    
                    fields_df = pd.DataFrame(fields_data)
                    data_frames[f'10_Form_{form_num}_Fields'] = fields_df
                
                # Sheet 3: Select Options (if any)
                select_options_data = []
                for field in fields:
                    if field.get('element_type') == 'select' and field.get('options'):
                        for opt in field.get('options', []):
                            select_options_data.append({
                                'Field_Name': field.get('name', ''),
                                'Field_ID': field.get('id', ''),
                                'Field_Label': field.get('label', ''),
                                'Option_Value': opt.get('value', ''),
                                'Option_Text': opt.get('text', ''),
                                'Selected': 'Yes' if opt.get('selected', False) else 'No',
                                'Disabled': 'Yes' if opt.get('disabled', False) else 'No'
                            })
                
                if select_options_data:
                    options_df = pd.DataFrame(select_options_data)
                    data_frames[f'10_Form_{form_num}_Options'] = options_df
        
        elif category == 'meta' and isinstance(items, list) and items:
            df = pd.DataFrame(items)
            df = df.fillna('')
            data_frames['11_Meta_Tags'] = df
    
    return data_frames

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze URL and return available content types"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': '📝 Please enter a URL to analyze'
            })
        
        scraper = WebScraper(url)
        
        # Validate URL
        is_valid, error_msg = scraper.validate_url()
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            })
        
        # Fetch page
        success, message = scraper.fetch_page()
        if not success:
            return jsonify({
                'success': False,
                'error': message
            })
        
        # Get available content
        content_analysis = scraper.get_available_content()
        
        if not content_analysis:
            return jsonify({
                'success': False,
                'error': '🔍 No scrapable content found on this page.'
            })
        
        return jsonify({
            'success': True,
            'available_content': content_analysis,
            'page_title': scraper.cleaner.clean_text(scraper.soup.find('title').string) if scraper.soup.find('title') else 'Unknown'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'⚠️ Analysis error: {str(e)}'
        })

@app.route('/scrape', methods=['POST'])
def scrape():
    """Scrape selected features - FULL scrape with no omissions"""
    try:
        data = request.json
        url = data.get('url', '').strip()
        selected_features = data.get('features', [])
        
        if not url:
            return jsonify({
                'success': False,
                'error': '📝 Please enter a URL to scrape'
            })
        
        scraper = WebScraper(url)
        
        # Validate URL
        is_valid, error_msg = scraper.validate_url()
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            })
        
        # Fetch page
        success, message = scraper.fetch_page()
        if not success:
            return jsonify({
                'success': False,
                'error': message
            })
        
        # Perform FULL scrape
        results = scraper.scrape_full(selected_features)
        
        # Count total items scraped
        total_items = sum(
            len(v) if isinstance(v, list) else 
            (sum(len(vv) if isinstance(vv, list) else 1 for vv in v.values()) if isinstance(v, dict) else 1)
            for v in results['data'].values()
        )
        
        if total_items == 0:
            return jsonify({
                'success': False,
                'error': '🔍 No data found. The page might be empty or JavaScript-heavy.'
            })
        
        return jsonify({
            'success': True,
            'data': results,
            'total_items': total_items,
            'message': f'✅ Successfully scraped {total_items} items!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'⚠️ Scraping error: {str(e)}'
        })

@app.route('/export', methods=['POST'])
def export_data():
    """Export data to Excel with clean, standardized format"""
    try:
        data = request.json
        export_format = data.get('format', 'xlsx')
        scrape_data = data.get('data', {})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format == 'json':
            output = io.BytesIO()
            output.write(json.dumps(scrape_data, indent=2, ensure_ascii=False).encode('utf-8'))
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/json',
                as_attachment=True,
                download_name=f'scraped_data_{timestamp}.json'
            )
        
        elif export_format in ['csv', 'xlsx']:
            # Create clean, organized data frames
            data_frames = create_clean_csv_data(scrape_data)
            
            if not data_frames:
                return jsonify({'success': False, 'error': 'No data to export'})
            
            # Create Excel file with multiple clean sheets
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in data_frames.items():
                    # Clean sheet name (Excel 31 char limit)
                    safe_sheet_name = sheet_name[:31]
                    
                    # Write to Excel
                    df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                    
                    # Auto-adjust column widths
                    worksheet = writer.sheets[safe_sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            output.seek(0)
            
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'scraped_data_{timestamp}.xlsx'
            )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Export error: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)