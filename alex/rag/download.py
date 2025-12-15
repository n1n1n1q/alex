import os
import sys
import zipfile

import requests
import wget


def get_doi(link: str) -> str:
    response = requests.get(link)
    return response.url


DOWNLOAD_URLS = {
    "full": f"{get_doi('https://doi.org/10.5281/zenodo.6640448')}/files/wiki_full.zip",
    "samples": f"{get_doi('https://doi.org/10.5281/zenodo.6640448')}/files/wiki_samples.zip",
}


def get_fn(download_dir: str, full: bool = True):
    url = DOWNLOAD_URLS["full" if full else "samples"]

    fn = wget.filename_from_url(url)          
    extracted_name = fn.replace(".zip", "")   

    download_fn = os.path.join(download_dir, fn)
    extracted_fn = os.path.join(download_dir, extracted_name)

    return extracted_fn, download_fn, url


def download(download_dir: str = None, full: bool = True) -> str:
    """Download MineDojo Wiki dataset.

    Args:
        download_dir: where to store data (default: ~/.minedojo)
        full: True -> wiki_full.zip, False -> wiki_samples.zip

    Returns:
        Path to extracted folder (e.g., ~/.minedojo/wiki_full or ~/.minedojo/wiki_samples)
    """
    if download_dir is None:
        download_dir = os.path.join(os.path.expanduser("~"), ".minedojo")
    os.makedirs(download_dir, exist_ok=True)

    extracted_fn, download_fn, url = get_fn(download_dir, full)

    
    if os.path.exists(extracted_fn):
        return extracted_fn

    
    if not os.path.exists(download_fn):
        print(f"Downloading {url} to {download_fn}...")
        wget.download(url, out=download_fn, bar=bar_progress)
        print()  

    
    with zipfile.ZipFile(download_fn, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    os.remove(download_fn)

    return extracted_fn


def bar_progress(current, total, width=80):
    progress_message = f"Downloading: {current / total * 100:.1f}% [{current/1e6:.1f} / {total/1e6:.1f}] MB"
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


download("/datasets/maksymz/alex/alex/rag/data")