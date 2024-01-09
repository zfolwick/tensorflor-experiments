#%%
# because it uses js to load the page, need to use selenium
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

def get_page_source(url):
	chrome_options = Options()
	chrome_options.add_argument('--headless')
	chrome_options.add_argument('--disable-gpu')
	driver = webdriver.Chrome(options=chrome_options)
	driver.get(url)
    # uncomment the below line to get a TON of data
	# driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	time.sleep(3)
	page = driver.page_source
	driver.quit()
	return page

#%%
# creates a list of threads on reddit.com/r/photoshopbattles.
import requests 
import re
from bs4 import BeautifulSoup 

def get_original_images():
	url = "https://www.reddit.com/r/photoshopbattles"
	page = get_page_source(url)

	soup = BeautifulSoup(page, 'html.parser') 
	anchors = soup.find_all('a')
	candidate_links = []
	for anchor in anchors:
		link = str(anchor.get('href'))
		pattern = r'^/r/photoshopbattles/comments/'

		if (re.search(pattern, link) and re.search(r'psbattle', link)):
			candidate_links.append(link)
	return list(set(candidate_links))

# for link in set(candidate_links):
#     print(link)
# %%
# take a candidate link and get all the photoshopped links
def get_modified_images(endpoint):
	new_page = get_page_source("https://www.reddit.com" + endpoint)
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

# %%
list_of_original_image_links = get_original_images()
print(list_of_original_image_links)

list_of_modified_image_links = {}
for link in list_of_original_image_links:
    list_of_modified_image_links[link] = get_modified_images(link)

# print { original: [... modified_links] }
for k,v in list_of_modified_image_links.items():
    print(f"{k} : {v}")
    
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
def save_image(url):
    filename = "./processing_queue/" + str(url).replace("https://","") + ".jpg"
    response = requests.get(url)
    if response.status_code == 200:
        # Open the image using Pillow
        img = Image.open(BytesIO(response.content))

        # Save the image as JPEG
        img.convert('RGB').save(filename, "JPEG")
        print(f"Image saved as {filename}")
#%%
# just a utility
import secrets
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string

# #%%
# # parse the modified and extract the actual image png or jpg.
# first_key, first_value = next(iter(list_of_modified_image_links.items()))
# # print(first_value)
# for modded in first_value:
# 	page = get_page_source(modded)
# 	actual_shopped_images_urls = get_actual_images(page)
# 	# save the list of shopped images to a directory
# 	for u in actual_shopped_images_urls:
# 		save_image(u, "./processing_queue/" + generate_random_string(8) + ".jpg")

# %%
