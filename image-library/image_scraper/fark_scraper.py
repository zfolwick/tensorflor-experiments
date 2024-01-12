#%%
import scraper as sc

#%%
# creates a list of threads on "https://www.fark.com/topic/photoshop".
import re
from bs4 import BeautifulSoup 

def get_original_images():
	url = "https://www.fark.com/topic/photoshop"
	page = sc.get_page_source(url)
 
	soup = BeautifulSoup(page, 'html.parser')
	anchors = soup.find_all('a')
	candidate_links = []
	for anchor in anchors:
		link = str(anchor.get('href'))
		pattern = r'fark\.com/comments/(\d{8})/Photoshop-this'
		
		if (re.search(pattern, link)):
			candidate_links.append(link)
	return list(set(candidate_links))

# %%
# take a candidate link and get all the photoshopped links
def get_modified_images(url):
	new_page = sc.get_page_source(url)
	soup = BeautifulSoup(new_page, 'html.parser') 
	shopped_anchors = soup.find_all('img')
 
	modified_links = []
	for anchor in shopped_anchors:
		link = str(anchor.get('data-src'))
		pattern = r'https://usrimg-850.fark.net'
		if (re.search(pattern, link)):
			modified_links.append(link.replace("amp;", ""))

	return modified_links[0], modified_links[1:]

#%%
# persist the images from a given url to disk under a filename
def save_image(classification, url):
    sc.save_image(classification, url)
        
def save_images(classification, urls):
    sc.save_images(classification, urls)
#%%
# just a utility
import secrets
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string

# %%
