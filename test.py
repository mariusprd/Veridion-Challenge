import requests
from time import sleep
import random
import pandas as pd
import numpy as np
import ssl

# Bypass SSL certificate verification if needed (trusted environments)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



# Define the path to the GloVe embeddings file
glove_file_path = 'glove.840B.300d.txt'

def load_embedding_model(file_path):
    """
    Load embeddings from a file.
    The file may optionally have a header line.
    Returns a dictionary mapping words to their vector embeddings.
    """
    model = {}
    expected_dim = None
    with open(file_path, 'r', encoding='utf-8') as f:
        # Check if the first line is a header (e.g., "400000 300")
        first_line = f.readline().strip().split()
        if len(first_line) == 2 and first_line[0].isdigit() and first_line[1].isdigit():
            expected_dim = int(first_line[1])
        else:
            # No header; reset file pointer
            f.seek(0)
        
        for line in f:
            if line.startswith('#'):
                continue  # skip comments
            values = line.strip().split()
            if len(values) < 2:
                continue
            word = values[0]
            try:
                vector = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue  # skip lines with conversion issues
            if expected_dim is None:
                expected_dim = len(vector)
            if len(vector) != expected_dim:
                print(f"Skipping {word} due to vector length mismatch: {len(vector)} vs {expected_dim}")
                continue
            model[word] = vector
    return model

print("Loading GloVe embeddings...")
glove_model = load_embedding_model(glove_file_path)
print(f"Total words loaded: {len(glove_model)}")

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def get_target_vector(target, model):
    """
    Retrieve a vector for a given target word/phrase.
    It tries several strategies: direct match, lower-casing,
    replacing spaces with underscores, and finally averaging tokens.
    """
    if target in model:
        return model[target]
    alt = target.replace(" ", "_")
    if alt in model:
        return model[alt]
    alt = target.lower()
    if alt in model:
        return model[alt]
    alt = target.lower().replace(" ", "_")
    if alt in model:
        return model[alt]
    
    # Fallback: split into tokens and average the vectors of tokens found in the model.
    tokens = target.split()
    vectors = []
    for token in tokens:
        if token in model:
            vectors.append(model[token])
        elif token.lower() in model:
            vectors.append(model[token.lower()])
    if vectors:
        return np.mean(vectors, axis=0)
    return None

def compare_word_to_set(new_word, word_set, model):
    """
    Compare the given new_word to each word in word_set using cosine similarity.
    
    Parameters:
      new_word (str): The word to compare.
      word_set (list of str): A list of words to compare against.
      model (dict): The pre-loaded embeddings model.
      
    Returns:
      dict: A dictionary mapping each word in word_set to its cosine similarity with new_word.
    """
    new_word_vec = get_target_vector(new_word, model)
    if new_word_vec is None:
        print(f"Word '{new_word}' not found in the model.")
        return {}
    
    similarities = {}
    for word in word_set:
        word_vec = get_target_vector(word, model)
        if word_vec is None:
            print(f"Word '{word}' not found in the model. Skipping.")
            continue
        sim = cosine_similarity(new_word_vec, word_vec)
        similarities[word] = sim
    return similarities

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5


def get_beat_dict():
    # this has the format {my_word: [the list of words that it beats]}
    return {
        "Feather": ["Heavy objects", "mass", "gravity", "inertia", "dense materials"],
        "Coal": ["Wood", "low-energy fuels", "biomass", "simple materials"],
        "Pebble": ["Fragile items", "delicate materials", "small obstacles", "surface materials"],
        "Leaf": ["Small debris", "dirt", "light materials", "organic matter", "plants"],
        "Paper": ["Fragile items", "light materials", "combustible materials", "trees"],
        "Rock": ["Fragile materials", "soft objects", "small structures", "softer stones", "hand"],
        "Water": ["Fire", "heat", "drought", "dehydration", "dry environments", "thirst", "dry", "sand"],
        "Twig": ["Small fragile things", "small animals", "leaves", "light structures", "insect"],
        "Sword": ["Humans", "animals", "unprotected bodies", "wood", "certain materials", "plants", "computer", "electronics", "hand", "foot"],
        "Shield": ["Physical attacks", "projectiles", "impact", "slashing weapons", "blunt force", "gunfire", "gun"],
        "Gun": ["Humans", "animals", "doors", "barriers", "weaker materials", "threats", "plants", "trees", "wood", "electronics", "computer"],
        "Flame": ["Flammable materials", "air", "fuel sources", "ice", "moisture", "cold", "house", "building", "structures", "wood", "paper", "plants", "trees", "animals", "humans", "living beings", "living creatures", "living things", "beings", "creatures", "things"],
        "Rope": ["Loose items", "unconnected materials", "smaller objects", "bodies", "insect"],
        "Disease": ["Living organisms", "immune systems", "human health", "animal populations", "human", "animal", "insect", "organisms", "beings", "creatures"],
        "Cure": ["Disease", "illness", "infections", "viral outbreaks", "wounds", "ailments"],
        "Bacteria": ["Weak organisms", "some immune systems", "low-level pathogens", "health", "clean", "sanitary", "hygiene"],
        "Shadow": ["Light", "sun", "illumination", "space", "bright areas", "mirror"],
        "Light": ["Darkness", "cold", "shadows", "low visibility", "obscured areas", "night", "absence of light"],
        "Virus": ["Living organisms", "immune systems", "cellular structures", "disease transmission"],
        "Sound": ["Silence", "stillness", "peace", "noise suppression", "disturbances", "quiet", "calm"],
        "Time": ["Everything", "decay", "memory", "civilization", "technology", "life cycles"],
        "Fate": ["Free will", "choices", "individual control", "randomness", "chaos", "war", "human", "animal", "plants", "life", "living", "organisms", "living beings", "living creatures", "living things", "beings", "creatures", "things"],
        "Earthquake": ["Buildings", "roads", "structures", "landscapes", "civilization", "foundations", "human", "animal", "plants", "life", "road", "bridge"],
        "Storm": ["Calm weather", "small plants", "fragile structures", "peace", "warm conditions", "fire", "animals", "plants"],
        "Vaccine": ["Disease", "viruses", "infections", "pathogens", "bacterial outbreaks"],
        "Logic": ["Emotion", "irrationality", "confusion", "errors in judgment", "subjective beliefs", "war", "conflict", "violence", "stress", "suffering", "unrest"],
        "Gravity": ["Objects in free-fall", "floating objects", "anti-gravity", "resistance", "levitation"],
        "Robots": ["Human labor", "manual tasks", "repetitive work", "hazardous tasks", "unskilled work"],
        "Stone": ["Fragile materials", "soft objects", "weaker structures", "easily destructible items"],
        "Echo": ["Silence", "stillness", "quiet environments", "soundless places"],
        "Thunder": ["Silence", "calm", "peaceful environments", "quiet places"],
        "Karma": ["Unjust actions", "bad behavior", "unethical decisions", "selfishness", "immorality"],
        "Wind": ["Still air", "calm weather", "fire", "smoke", "stagnant environments"],
        "Ice": ["Fire", "heat", "high temperatures", "warm environments", "liquid water"],
        "Sandstorm": ["Calm weather", "clear skies", "small", "fragile objects", "stillness"],
        "Laser": ["Obstacles", "barriers", "physical defenses", "darkness", "resistance"],
        "Magma": ["Structures", "buildings", "ice", "soil", "vegetation", "cold weather"],
        "Peace": ["War", "conflict", "violence", "stress", "suffering", "unrest"],
        "Explosion": ["Fragile structures", "fragile materials", "stability", "unprotected buildings", "people"],
        "War": ["Peace", "diplomacy", "calm", "stability", "safety", "serenity"],
        "Enlightenment": ["Ignorance", "confusion", "limited knowledge", "darkness", "misinformation"],
        "Nuclear Bomb": ["Buildings", "infrastructure", "large areas", "human life", "nature"],
        "Volcano": ["Structures", "landscapes", "vegetation", "human settlements", "ecosystems"],
        "Whale": ["Smaller sea creatures", "fish", "sea life", "ocean predators"],
        "Earth": ["Artificial objects", "space debris", "cosmic matter", "asteroids"],
        "Moon": ["Light", "planets", "stars", "darkness", "celestial bodies"],
        "Star": ["Darkness", "cold", "void", "empty space", "night", "absence of light"],
        "Tsunami": ["Coastal structures", "land", "buildings", "small islands", "shorelines"],
        "Supernova": ["Stars", "planets", "systems", "matter", "space debris"],
        "Antimatter": ["Matter", "physical objects", "standard matter", "material substances"],
        "Plague": ["Living organisms", "cities", "civilizations", "human health", "animal populations"],
        "Rebirth": ["Death", "decay", "stagnation", "endings", "loss"],
        "Tectonic Shift": ["Landscapes", "human structures", "geological formations", "topography", "house", "building", "structures"],
        "Gamma-Ray Burst": ["Stars", "matter", "life", "biological organisms", "planetary systems", "life", "human", "animal", "plant", "organism"],
        "Human Spirit": ["Suffering", "negative emotions", "hardships", "despair", "hopelessness", "war", "conflict", "violence", "stress", "suffering", "unrest"],
        "Apocalyptic Meteor": ["Civilization", "life", "structures", "ecosystems", "atmosphere", "human", "animal", "plants", "life", "living", "organisms", "living beings", "living creatures", "objects", "things"],
        "Earth’s Core": ["Surface", "outer layers", "land", "mineral formations", "structures", "human", "animal", "music instrument", "broom", "spoon", "household", "objects"],
        "Neutron Star": ["Ordinary matter", "space", "light matter", "stellar systems", "stars", "planets", "human", "animal", "plant", "organism", "building", "structure", "household", "objects", "spoon", "broom", "music instrument"],
        "Supermassive Black Hole": ["Stars", "galaxies", "space", "matter", "cosmic structures", "celestial bodies", "human life", "human", "animal", "plants", "life", "living", "organisms", "living beings", "living creatures", "living things", "beings", "creatures", "things", "buildings", "structures"],
        "Entropy": ["Order", "systems", "organized structures", "predictability", "uniformity", "stability", "harmony", "balance"]
    }

# function that takes a dictionary that maps a word to a list then returns all the lists concatenated
def get_all_words(beat_dict):
    all_words = []
    for words in beat_dict.values():
        all_words.extend(word.lower() for word in words)
    return all_words

def get_key_from_value(value, beat_dict):
    # Iterate through the dictionary
    for key, values in beat_dict.items():
        # Check if the value is in the list of values for the current key
        if value in values:
            return key
    return None  # Return None if not found

def read_words():
    file_path = "my_words.csv"
    try:
        df = pd.read_csv(file_path)
        if 'cost' in df.columns:
            df = df.sort_values(by='cost', ascending=True)  # Order by cost from lowest to highest
        return df
    except Exception as e:
        print(f"Error reading words from {file_path}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame as a fallback


def backup_word():
    global my_words
    if not my_words.empty:
        return int(my_words.index[1])  # Return the index of the first word as an integer
    else:
        print("No words available to backup.")
        return 1



def what_beats(word):
    # get closest words from our list for the given word

    words_to_compare = get_all_words(beat_dict)
    # get the vector for the word
    word_vec = get_target_vector(word.lower(), glove_model)
    # get the cosine similarity for each word
    similarities = compare_word_to_set(word.lower(), words_to_compare, glove_model)
    # get top 5 closes words
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_5_closest = sorted_similarities[:5]
    # print them
    print(f"Top 5 closest words to '{word}':")
    for word, sim in top_5_closest:
        print(f"{word}: {sim:.4f}")

    # get list of words that beat the given word
    beat_words = []
    for key, values in beat_dict.items():
        for closest_word, _ in top_5_closest:
            if closest_word in values:
                beat_words.append(key)
    # print them
    print(f"Words that beat '{word}': {beat_words}")

    # choose the one with the lowest cost?..
    #choose the first word then return its id

    if beat_words:
        # get the first word from the list
        chosen_word = beat_words[0]
        # get the id and cost of the word from my_words
        chosen_word_row = my_words.loc[my_words['word'].str.lower() == chosen_word.lower()]
        if not chosen_word_row.empty:
            chosen_word_id = chosen_word_row['id'].values[0]
            chosen_word_cost = chosen_word_row['cost'].values[0]
            if chosen_word_cost > 32:
                print(f"Cost of '{chosen_word}' is too high ({chosen_word_cost}). Sleeping instead.")
                sleep(1)
                return 1
            print(f"Chosen word: '{chosen_word}' with ID: {chosen_word_id} and cost: {chosen_word_cost}")
            return int(chosen_word_id)

    return 1


def play_game(player_id):

    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

            sleep(1)

        if round_id > 1:
            status = requests.get(status_url)
            print(status.json())


        # choose backup word
        choosen_word = backup_word()
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(f"Submitted backup word: {choosen_word} for round {round_id}")
        print(response.json())


        # choose actual word
        choosen_word = what_beats(sys_word)
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(f"Submitted actual word: {choosen_word} for round {round_id}")
        print(response.json())


# player_id = "ceva"
# print("Starting game...")
# play_game(player_id)

my_words = read_words()
print(f"Finished reading my words\nHere are the first 5:\n{my_words.head(5)}\n")

beat_dict = get_beat_dict()


player_id = 'a simple quick brown fox'
print(f"Starting game for player ID: {player_id}")
play_game(player_id)
print("Game finished.")