import scraper as sc
#%%
# because it uses js to load the page, need to use selenium
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


#%%
# creates a list of threads on reddit.com/r/photoshopbattles.
import requests 
import re
from bs4 import BeautifulSoup 

def get_original_images():
    url = "https://www.reddit.com/r/photoshopbattles"
    page = sc.get_page_source(url)

    soup = BeautifulSoup(page, 'html.parser') 
    anchors = soup.find_all('a')
    candidate_links = []
    for anchor in anchors:
        link = str(anchor.get('href'))
        pattern = r'^/r/photoshopbattles/comments/'

        if (re.search(pattern, link) and re.search(r'psbattle', link)):
           
            candidate_links.append(f"https://reddit.com{link}")
    return list(set(candidate_links))

# %%
# take a candidate link and get all the photoshopped links
def get_modified_images(endpoint):
	new_page = sc.get_page_source("https://www.reddit.com" + endpoint)
	soup = BeautifulSoup(new_page, 'html.parser') 
	shopped_anchors = soup.find_all('a')
	modified_links = []
	for anchor in shopped_anchors:
		link = str(anchor.get('href'))
		# print(link)
		pattern = r'imgur'
		if (re.search(pattern, link)):
			modified_links.append(link)

	return modified_links

#%%
# get the actual images
def get_actual_images(page):
	soup = BeautifulSoup(page, 'html.parser') 
	shopped_images = soup.find_all('img')
	shopped_image_assets = []
	for i in shopped_images:
		current = str(i)
		if ('imgur' in current and not 'desktop-assets' in current and not 'favicon' in current):
			shopped_image_assets.append(i['src'].split('?')[0])
	return list(set(shopped_image_assets))

#%%
# persist the images from a given url to disk under a filename
import requests
from PIL import Image
from io import BytesIO
def save_image(classification, url):
    sc.save_image(classification, url)
        
def save_images(classification, urls):
    sc.save_images(classification, url)
#%%
# just a utility
import secrets
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string

# %%
