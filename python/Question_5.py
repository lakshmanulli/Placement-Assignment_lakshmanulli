import requests

def download_data(url):
    response = requests.get(url)  # Send a GET request to the URL
    data = response.json()  # Get the JSON data from the response
    return data

def extract_show_data(data):
    show_id = data.get('id', '')
    show_url = data.get('url', '')
    show_name = data.get('name', '')
    show_type = data.get('type', '')
    show_summary = data.get('summary', '').replace('<p>', '').replace('</p>', '')
    
    image_data = data.get('image', {})
    image_medium = image_data.get('medium', '')
    image_original = image_data.get('original', '')
    
    show_data = {
        'id': show_id,
        'url': show_url,
        'name': show_name,
        'type': show_type,
        'summary': show_summary,
        'medium_image': image_medium,
        'original_image': image_original
    }
    
    return show_data

def extract_episode_data(data):
    episodes = data.get('_embedded', {}).get('episodes', [])
    
    episode_data = []
    for episode in episodes:
        episode_id = episode.get('id', '')
        episode_season = episode.get('season', '')
        episode_number = episode.get('number', '')
        episode_airdate = episode.get('airdate', '')
        episode_airtime = episode.get('airtime', '')
        episode_runtime = episode.get('runtime', '')
        episode_rating = episode.get('rating', {}).get('average', '')
        episode_summary = episode.get('summary', '').replace('<p>', '').replace('</p>', '')
        
        episode_entry = {
            'id': episode_id,
            'season': episode_season,
            'number': episode_number,
            'airdate': episode_airdate,
            'airtime': episode_airtime,
            'runtime': episode_runtime,
            'average_rating': episode_rating,
            'summary': episode_summary
        }
        
        episode_data.append(episode_entry)
    
    return episode_data

# Download the data from the provided API link
url = 'http://api.tvmaze.com/singlesearch/shows?q=westworld&embed=episodes'
data = download_data(url)

# Extract the show data and episode data with proper formatting
show_data = extract_show_data(data)
episode_data = extract_episode_data(data)

# Print the extracted show data
print("Show Data:")
print("---------")
for key, value in show_data.items():
    print(f"{key}: {value}")
print()

# Print the extracted episode data
print("Episode Data:")
print("------------")
for episode in episode_data:
    print("Episode:")
    for key, value in episode.items():
        print(f"{key}: {value}")
    print()