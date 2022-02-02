import requests

download_path = "data_downloadable"


def download_from_url(url, file_name):
    with requests.get(url, stream=True) as r:
        with open(f"{download_path}/{file_name}", "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"{file_name} download complete")
