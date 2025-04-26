import os
import requests
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def search_images(query, max_results=20):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.images(keywords=query, max_results=max_results):
            results.append(r)
    return results

def download_and_classify_images(query, output_base_dir, max_images=20):
    class_dirs = {
        '1': os.path.join(output_base_dir, 'Cat'),
        '2': os.path.join(output_base_dir, 'Dog'),
    }
    
    for path in class_dirs.values():
        ensure_dir(path)
        
    results = search_images(query, max_results=max_images)

    for idx, result in enumerate(results):
        url = result['image']
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img.show()

            choice = input("Classify this image (0 = skip, 1 = Cat, 2 = Dog): ").strip()

            if choice in ['1', '2']:
                save_path = os.path.join(class_dirs[choice], f"{query.replace(' ', '_')}_{idx}.jpg")
                img.save(save_path)
                print(f"Saved to {save_path}\n")
            else:
                print("Skipped.\n")

        except Exception as e:
            print(f"Failed to process image {idx}: {e}\n")

if __name__ == "__main__":
    query = input("Enter search query: ")
    output_base_dir = input("Enter base output directory (default: './data'): ") or './data'
    max_images = int(input("Enter max number of images to scrape (default: 100): ") or 100)

    download_and_classify_images(query, output_base_dir, max_images)
