import reddit_scraper as rs

print(">> get some original image links")
original_image_links = rs.get_original_images()
print(">> SANITY CHECK: list of links to images on the front page")
print(original_image_links)
#TODO: persist the original images to disk

#for each original image, visit that page and gather the potentially modified images
list_of_modified_image_links = {}
for link in original_image_links:
    list_of_modified_image_links[link] = rs.get_modified_images(link)

print("SANITY CHECK: original image link + list of possible modded links")
# print { original: [... modified_links] }
for k,v in list_of_modified_image_links.items():
    print(f"{k} : {v}")
    

#from the list of possibly modded links, get the actual image, and not a page with other stuff on it.
for k, v in list_of_modified_image_links.items():
    print("getting shoops for " + k)
    # TODO: handle original (key) - persist image to disk
    # get actual images from modded link
    for modded in v:
        print("possible mod: " + modded)
        page = rs.get_page_source(modded)
        actual_shopped_images_urls = rs.get_actual_images(page)
        # save the list of shopped images to processing_queue directory
        for u in actual_shopped_images_urls:
            rs.save_image(u)
