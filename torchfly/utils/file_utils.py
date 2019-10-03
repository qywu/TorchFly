import requests
import tqdm

def http_get(url, filename, proxies=None):
    
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm.tqdm(unit="B", total=total)
    
    with open(filename, "wb") as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                f.write(chunk)
            
    progress.close()