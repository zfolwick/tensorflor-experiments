import Loader as ld

loader = ld.Loader("./image-library")
# loader.purpose()
# loader.displayFirstImage("modified/*", 0)
# loader.displayFirstImage("original/*", 0)
loader.load()
loader.visualize()
print("breakpoint")