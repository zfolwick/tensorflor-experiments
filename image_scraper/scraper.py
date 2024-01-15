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
    chrome_options.add_argument('--log-level=3')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    # uncomment the below line to get a TON of data
    # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)
    page = driver.page_source
    driver.quit()
    return page

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
# Files in windows OS cannot have "?", "&" or superfluous ".", and "/" just messes things up.
def formatForFs(url):
        return str(url).replace("https://", "").replace("?", "@").replace("&", "^").replace("/", "#")

#%%
# persist the images from a given url to disk under a filename
import requests
from PIL import Image
from io import BytesIO
import os
def save_image(classification, originalCounter, url):
    print(url)
    dirname = f"processing_queue/{classification}/{originalCounter}"
    isExist = os.path.exists(dirname)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dirname)
    filename = f"{dirname}/" + formatForFs(url) + ".jpg"
    print(f"Saving file {filename}")
    response = requests.get(url)
    if response.status_code == 200:
        # Open the image using Pillow
        try:
            img = Image.open(BytesIO(response.content))
            # Save the image as JPEG
            img.convert('RGB').save(filename, "JPEG")
        except Exception as e:
            print(f"there was an issue saving {filename}")
            print(f"the source url is: {url}")
            print(e)
            return
        print(f"Image saved as {filename}")
        
def save_original_image(originalCounter, url):
    dirname = f"processing_queue/original"
    isExist = os.path.exists(dirname)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dirname)
    filename = f"{dirname}/{originalCounter}.jpg"
    print(f"Saving file {filename}")
    response = requests.get(url)
    if response.status_code == 200:
        # Open the image using Pillow
        try:
            img = Image.open(BytesIO(response.content))
            # Save the image as JPEG
            img.convert('RGB').save(filename, "JPEG")
        except Exception as e:
            print(f"there was an issue saving {filename}")
            print(f"the source url is: {url}")
            print(e)
            return
        print(f"Image saved as {filename}")
        
def save_images(classification, originalCounter, urls):
    for url in urls:
        save_image(classification, originalCounter, url)
#%%
# just a utility
import secrets
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string
#%%
