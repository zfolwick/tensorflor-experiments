#%%
# The below code runs returns the url for all img tags.
import requests 
from bs4 import BeautifulSoup 
	
def getdata(url): 
	r = requests.get(url) 
	return r.text 
	
htmldata = getdata("https://www.fark.com/") 
soup = BeautifulSoup(htmldata, 'html.parser') 
for item in soup.find_all('img'): 
	print(item['src'])
 
#%%
# This part creates a list of threads on fark.com.
import requests 
import re
from bs4 import BeautifulSoup 
	
def getdata(url): 
	r = requests.get(url) 
	return r.text 
	
htmldata = getdata("https://www.fark.com/topic/photoshop") 
soup = BeautifulSoup(htmldata, 'html.parser') 
anchors = soup.find_all('a')

candidate_links = []
for anchor in anchors:
    link = str(anchor.get('href'))
    pattern = r'fark\.com/comments/(\d{8})'
    
    if (re.search(pattern, link)):
        candidate_links.append(link)

# print(candidate_links)
thread = candidate_links[0]
print(">>>>" + thread)
#%%
htmldata2 = getdata(thread) 
soup2 = BeautifulSoup(htmldata2, 'html.parser') 
anchors2 = soup2.find_all('img')
# print(anchors2)
for a in anchors2:
	if 'data-origsrc' in str(a):
		print(a)


# %%
