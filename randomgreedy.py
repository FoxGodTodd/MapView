#random greedy
import numpy as np
import pandas as pd
from geopy.distance import geodesic

def addRandomSwaps(df,ordered_indices):
    order = np.copy(ordered_indices)
    M = len(order)
    bestdist = float('inf')
    for trial in range(M*(M-1)*2):
        totdist = 0
        for v in range(len(order)-1):
            dist = 0
            lat1, lon1 = df.loc[order[v], 'Latitude'], df.loc[order[v], 'Longitude']
            lat2, lon2 = df.loc[order[v+1], 'Latitude'], df.loc[order[v+1], 'Longitude']
            dist = calculate_distance(lat1, lon1, lat2, lon2)
            if v >= 1:
                prev_index = order[v-1]
                prev_lat, prev_lon = df.loc[prev_index, 'Latitude'], df.loc[prev_index, 'Longitude']
                #prev_lat, prev_lon = df.loc[prev_index, 'coordinates']
                v1 = np.array([lat1 - prev_lat, lon1 - prev_lon])
                v2 = np.array([lat2 - lat1, lon2 - lon1])
                angle = 180*calculate_angle(v1, v2)/np.pi
                dist+=(angle/M) #bigger angles mean worse penalties
            totdist += dist
        if totdist < bestdist:
            bestdist = totdist
            better_indices = order
        order = swap_random(ordered_indices)
    return(better_indices)

def swap_random(seq):
    LEN = len(seq)
    neq = np.copy(seq)
    a, b = np.random.randint(LEN, size=2)
    neq[a], neq[b] = seq[b], seq[a]
    return(list(neq))
    
# Function to calculate distance between two points (latitude, longitude)
def calculate_distance(lat1, lon1, lat2, lon2):
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    return geodesic(coords_1, coords_2,ellipsoid='Airy (1830)').km
    
# Function to calculate the angle between two vectors (used to detect backtracking)
def calculate_angle(v1, v2):
    # Dot product and magnitudes
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    # Angle in radians
    angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    
    return angle

# Function to calculate penalty based on the angle
def backtrack_penalty(v1, v2,randfloat1):
    # If the angle is too sharp (i.e., backtracking), add penalty
    angle = calculate_angle(v1, v2)
    # Apply a penalty for angles greater than a threshold (e.g., 120 degrees)
    if angle > np.pi:  # 60 - 180 degrees in radians (for sharper turns)
        #return(1.0)
        return(2.5+randfloat1.random())  # Penalty factor
    return 1.0  # No penalty for smoother transitions

# Modified Greedy Algorithm with Random Sampling (Start from any point)
def greedy_with_random_sampling(df,randfloat1):
    n = len(df)
    print(n)
    #num_samples = int(n/2)
    best_order = None
    best_distance = float('inf')

    for m in range(n):
        # Randomly select a starting point
        start_index = m
        visited = [False] * n
        ordered_indices = [start_index]
        visited[start_index] = True
        current_index = start_index
        total_distance = 0
        
        # Greedy path search with backtracking penalty and lookahead
        for _ in range(1, n):
            closest_index = None
            min_distance = float('inf')

            # Look-ahead strategy: consider next two unvisited points
            possible_paths = []

            for i in range(n):
                if not visited[i]:
                    # Calculate distance for paths
                    lat1, lon1 = df.loc[current_index, 'Latitude'], df.loc[current_index, 'Longitude']
                    lat2, lon2 = df.loc[i, 'Latitude'], df.loc[i, 'Longitude'],
                    #lat1, lon1 = df.loc[current_index, 'coordinates']
                    #lat2, lon2 = df.loc[i, 'coordinates']
                    distance = calculate_distance(lat1, lon1, lat2, lon2)
                    
                    # Add backtrack penalty if applicable
                    if len(ordered_indices) >= 2:
                        prev_index = ordered_indices[-2]
                        prev_lat, prev_lon = df.loc[prev_index, 'Latitude'], df.loc[prev_index, 'Longitude']
                        #prev_lat, prev_lon = df.loc[prev_index, 'coordinates']
                        
                        # Vectors from previous to current, and from current to next candidate
                        v1 = np.array([lat1 - prev_lat, lon1 - prev_lon])
                        v2 = np.array([lat2 - lat1, lon2 - lon1])
                        
                        # Calculate penalty based on the direction change (backtracking)
                        penalty = backtrack_penalty(v1, v2, randfloat1)
                        distance *= penalty
                    
                    # Check if this point gives a better path
                    if distance < min_distance:
                        min_distance = distance
                        closest_index = i
            
            visited[closest_index] = True
            ordered_indices.append(closest_index)
            total_distance += min_distance
            current_index = closest_index
        
        # Check if this path is the best one found so far
        print(total_distance,ordered_indices)
        if total_distance < best_distance:
            best_distance = total_distance
            best_order = ordered_indices

    return best_order

def mainloop(uploadFile):
    randfloat1 = np.random.default_rng() 
    df = pd.read_excel(uploadFile)
    df2 = pd.read_csv('https://raw.githubusercontent.com/FoxGodTodd/MapView/main/Locations.csv')
    dforiginal = df.copy()
    
    df = df.dropna(subset=['Postcode'],ignore_index=True)  # Remove NaNs
    df2 = df2.dropna(subset=['postCode'],ignore_index=True)  # Remove NaNs
    
    df = df[df['Postcode'] != '']       # Remove empty strings
    df = df.drop_duplicates(subset='Postcode',ignore_index=True)  # Remove duplicates

    # Create new columns for Latitude and Longitude in df
    df['Latitude'] = np.nan
    df['Longitude'] = np.nan

    # Iterate through rows in df
    for index, row in df.iterrows():
        postcode = row['Postcode']
    
        # Filter rows in df2 that match the postcode
        matching_rows = df2[df2['postCode'] == postcode]
         
        if matching_rows.empty:
            postcode = postcode[:postcode.find(' ')]
            matching_rows = df2[df2['postCode'].str.contains(postcode)]
        if not matching_rows.empty:
            # Select a random row from the matching rows
            random_row = matching_rows.sample(n=1)
            df.at[index, 'Latitude'] = random_row['latitude'].values[0]
            df.at[index, 'Longitude'] =  random_row['longitude'].values[0] 
        else:
            print(postcode)     
    print(df)
    df = df.dropna(subset=['Latitude'],ignore_index=True)  # Remove NaNs
    
    ordered_indices = greedy_with_random_sampling(df,randfloat1)
    ordered_indices = addRandomSwaps(df,ordered_indices)
    reordered_df = df.iloc[ordered_indices].reset_index(drop=True)
    
    postcode_to_letter = {postcode: letter for postcode, letter in zip(reordered_df['Postcode'].unique(), string.ascii_uppercase)}
    dforiginal['Map'] = dforiginal['Postcode'].map(postcode_to_letter)
    dforiginal = dforiginal.set_index('Map')
    
    reordered_df['Map'] = list(string.ascii_uppercase[:len(reordered_df)])
    reordered_df.to_excel('reorderedcoords.xlsx', index=False)
    reordered_df = reordered_df.set_index('Map')  
    
    latitudes = reordered_df['Latitude'].to_numpy()
    longitudes = reordered_df['Longitude'].to_numpy()
    combined = []

    for lat, lon in zip(latitudes, longitudes):
        combined_string = f"{lat},{lon}!"
        combined.append(combined_string)
        final_string = ''.join(combined)
    mapstring = 'https://maps.tripomatic.com/#/?map=11,'
    url=mapstring+combined[0][:-1]+'&route='+final_string[:-1]+'&mode=car'
   
    return(reordered_df,dforiginal,url)
