import requests

def main():
    url = "https://vora-52k9.onrender.com/metrics"
    print(f"Querying Render metrics from {url}...")
    try:
        res = requests.get(url, timeout=10)
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.text}")
    except Exception as e:
        print(f"Error querying endpoint: {e}")

if __name__ == "__main__":
    main()
