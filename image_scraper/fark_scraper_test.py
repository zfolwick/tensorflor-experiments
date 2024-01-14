#%%
import fark_scraper as fs

print(">> get some original image links")
original_image_links = fs.get_original_images()
# print(">> SANITY CHECK: list of links to images on the front page")
# print(original_image_links)   
# print(">> SANITY CHECK: list of links to images on the front page")
#%%
#TODO: persist the original images to disk

# For each original link, visit that page and gather the potentially modified images. 
# The first will always be the original.
list_of_modified_image_links = {}
for link in original_image_links:
    original_image_link, modified_image_links = fs.get_modified_images(link)
    list_of_modified_image_links[original_image_link] = modified_image_links
#%%
print("SANITY CHECK: original image link + list of possible modded links")
# print { original1: [... modified_links], original2: [... modified_links] }
for original,modded_links in list_of_modified_image_links.items():
    print(f"{original} : {modded_links}")
    
#%%
#from the list of possibly modded links, get the actual image, and not a page with other stuff on it.
for original, modded_links in list_of_modified_image_links.items():
    print("getting images for " + original)                                                             
    fs.save_image("../image-library/original", original)
    fs.save_images("../image-library/modified", modded_links)
