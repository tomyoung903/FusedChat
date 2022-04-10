"""
"""
import json
import os
import random
from fuzzywuzzy import fuzz
from nltk.stem.porter import *

stemmer = PorterStemmer()

# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
dbs = {}
for domain in domains:
    dbs[domain] = json.load(open('multiwoz/db/{}_db.json'.format(domain)))

FUZZY_KEYS = {
    'hotel' : {'name'},
    'attraction' : {'name'},
    'restaurant' : {'name', 'food'},
    'train' : {'departure', 'destination'}
}


def normalize_state_slot_value(slot_name, value):
    """ Normalize slot value:
        1) replace too distant venue names with canonical values
        2) replace too distant food types with canonical values
        3) parse time strings to the HH:MM format
        4) resolve inconsistency between the database entries and parking and internet slots
    """
    
    def type_to_canonical(type_string): 
        if type_string == "swimming pool":
            return "swimmingpool" 
        elif type_string == "mutliple sports":
            return "multiple sports"
        elif type_string == "night club":
            return "nightclub"
        elif type_string == "guest house":
            return "guesthouse"
        return type_string

    def name_to_canonical(name, domain=None):
        """ Converts name to another form which is closer to the canonical form used in database. """

        name = name.strip().lower()
        name = name.replace(" & ", " and ")
        name = name.replace("&", " and ")
        name = name.replace(" '", "'")

        if domain is None or domain == "restaurant":
            if name == "hotel du vin bistro":
                return "hotel du vin and bistro"
            elif name == "the river bar and grill":
                return "the river bar steakhouse and grill"
            elif name == "nando's":
                return "nandos"
            elif name == "city center b and b":
                return "city center north b and b"
            elif name == "acorn house":
                return "acorn guest house"
            elif name == "caffee uno":
                return "caffe uno"
            elif name == "cafe uno":
                return "caffe uno"
            elif name == "rosa's":
                return "rosas bed and breakfast"
            elif name == "restaurant called two two":
                return "restaurant two two"
            elif name == "restaurant 2 two":
                return "restaurant two two"
            elif name == "restaurant two 2":
                return "restaurant two two"
            elif name == "restaurant 2 2":
                return "restaurant two two"
            elif name == "restaurant 1 7" or name == "restaurant 17":
                return "restaurant one seven"

        if domain is None or domain == "hotel":
            if name == "lime house":
                return "limehouse"
            elif name == "cityrooms":
                return "cityroomz"
            elif name == "whale of time":
                return "whale of a time"
            elif name == "huntingdon hotel":
                return "huntingdon marriott hotel"
            elif name == "holiday inn exlpress, cambridge":
                return "express by holiday inn cambridge"
            elif name == "university hotel":
                return "university arms hotel"
            elif name == "arbury guesthouse and lodge":
                return "arbury lodge guesthouse"
            elif name == "bridge house":
                return "bridge guest house"
            elif name == "arbury guesthouse":
                return "arbury lodge guesthouse"
            elif name == "nandos in the city centre":
                return "nandos city centre"

        if domain is None or domain == "attraction":
            if name == "broughton gallery":
                return "broughton house gallery"
            elif name == "scudamores punt co":
                return "scudamores punting co"
            elif name == "cambridge botanic gardens":
                return "cambridge university botanic gardens"
            elif name == "the junction":
                return "junction theatre"
            elif name == "trinity street college":
                return "trinity college"
            elif name in ['christ college', 'christs']:
                return "christ's college"
            elif name == "history of science museum":
                return "whipple museum of the history of science"
            elif name == "parkside pools":
                return "parkside swimming pool"
            elif name == "the botanical gardens at cambridge university":
                return "cambridge university botanic gardens"
            elif name == "cafe jello museum":
                return "cafe jello gallery"

        return name

    def time_to_canonical(time):
        """ Converts time to the only format supported by database, e.g. 07:15. """
        time = time.strip().lower()

        if time == "afternoon": return "13:00"
        if time == "lunch" or time == "noon" or time == "mid-day" or time == "around lunch time": return "12:00"
        if time == "morning": return "08:00"
        if time.startswith("one o'clock p.m"): return "13:00"
        if time.startswith("ten o'clock a.m"): return "10:00"
        if time == "seven o'clock tomorrow evening":  return "07:00"
        if time == "three forty five p.m":  return "15:45"
        if time == "one thirty p.m.":  return "13:30"
        if time == "six fourty five":  return "06:45"
        if time == "eight thirty":  return "08:30"

        if time.startswith("by"):
            time = time[3:]

        if time.startswith("after"):
            time = time[5:].strip()

        if time.startswith("afer"):
            time = time[4:].strip()    

        if time.endswith("am"):   time = time[:-2].strip()
        if time.endswith("a.m."): time = time[:-4].strip()

        if time.endswith("pm") or time.endswith("p.m."):
            if time.endswith("pm"):   time = time[:-2].strip()
            if time.endswith("p.m."): time = time[:-4].strip()
            tokens = time.split(':')
            if len(tokens) == 2:
                return str(int(tokens[0]) + 12) + ':' + tokens[1] 
            if len(tokens) == 1 and tokens[0].isdigit():
                return str(int(tokens[0]) + 12) + ':00'
        
        if len(time) == 0:
            return "00:00"
            
        if time[-1] == '.' or time[-1] == ',' or time[-1] == '?':
            time = time[:-1]
            
        if time.isdigit() and len(time) == 4:
            return time[:2] + ':' + time[2:]

        if time.isdigit(): return time.zfill(2) + ":00"
        
        if ':' in time:
            time = ''.join(time.split(' '))

        if len(time) == 4 and time[1] == ':':
            tokens = time.split(':')
            return tokens[0].zfill(2) + ':' + tokens[1]

        return time

    def food_to_canonical(food):
        """ Converts food name to caninical form used in database. """

        food = food.strip().lower()

        if food == "eriterean": return "mediterranean"
        if food == "brazilian": return "portuguese"
        if food == "sea food": return "seafood"
        if food == "portugese": return "portuguese"
        if food == "modern american": return "north american"
        if food == "americas": return "north american"
        if food == "intalian": return "italian"
        if food == "italain": return "italian"
        if food == "asian or oriental": return "asian"
        if food == "english": return "british"
        if food == "australasian": return "australian"
        if food == "gastropod": return "gastropub"
        if food == "brutish": return "british"
        if food == "bristish": return "british"
        if food == "europeon": return "european"

        return food

    if slot_name in ["name", "destination", "departure"]:
        return name_to_canonical(value)
    elif slot_name == "type":
        return type_to_canonical(value)
    elif slot_name == "food":
        return food_to_canonical(value)
    elif slot_name in ["arrive", "leave", "arriveby", "leaveat", "time"]:
        return time_to_canonical(value)
    elif slot_name in ["parking", "internet"]:
        return "yes" if value == "free" else value
    else:
        return value


def query(domain, constraints, ignore_open=True):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""

    # query the db
    if domain == 'taxi':
        return [{'taxi_colors': random.choice(dbs[domain]['taxi_colors']), 
        'taxi_types': random.choice(dbs[domain]['taxi_types']), 
        'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
    if domain == 'police':
        return dbs['police']
    if domain == 'hospital':
        return dbs['hospital']

    found = []
    for i, record in enumerate(dbs[domain]):
        for key, val in constraints:
            if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                pass
            else:
                try:
                    record_keys = [key.lower() for key in record]
                    if key.lower() not in record_keys and stemmer.stem(key) not in record_keys:
                        continue
                    if key == 'leaveAt':
                        val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                        val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                        if val1 > val2:
                            break
                    elif key == 'arriveBy':
                        val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                        val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                        if val1 < val2:
                            break
                    # elif ignore_open and key in ['destination', 'departure', 'name']:
                    elif ignore_open and key in ['destination', 'departure']:
                        continue
                    else:
                        if val.strip() != record[key].strip():
                            break
                except:
                    continue
        else:
            record['Ref'] = f'{i:08d}'
            found.append(record)

    return found


def query_fuzzy_and_normalized(domain, constraints, ignore_open=True):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""

    # query the db
    if domain == 'taxi':
        return [{'taxi_colors': random.choice(dbs[domain]['taxi_colors']), 
        'taxi_types': random.choice(dbs[domain]['taxi_types']), 
        'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
    if domain == 'police':
        return dbs['police']
    if domain == 'hospital':
        return dbs['hospital']

    found = []
    for i, record in enumerate(dbs[domain]):
        for key, val in constraints:
            if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                pass
            else:
                try:
                    record_keys = [key.lower() for key in record]
                    if key.lower() not in record_keys and stemmer.stem(key) not in record_keys:
                        continue
                    if key == 'leaveAt':
                        val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                        val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                        if val1 > val2:
                            break
                    elif key == 'arriveBy':
                        val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                        val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                        if val1 < val2:
                            break
                    # elif ignore_open and key in ['destination', 'departure', 'name']:
                    elif ignore_open and key in ['destination', 'departure']:
                        continue
                    else:
                        val = normalize_state_slot_value(key.lower().strip(), val)
                        if key.lower().strip() in FUZZY_KEYS.get(domain, {}):
                            f = (lambda x: fuzz.partial_ratio(record[key].strip(), x) < 90)
                        else:
                            f = (lambda x: record[key].strip() != x)
                        if f(val):
                            break
                        # if val.strip() != record[key].strip():
                        #     break
                except:
                    continue
        else:
            record['Ref'] = f'{i:08d}'
            found.append(record)

    return found

