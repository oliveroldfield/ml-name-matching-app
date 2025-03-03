import pandas as pd
import random
import numpy as np

# Generate sample training data
def generate_sample_data(n_samples=1000, match_ratio=0.5, include_name_order_swap=True):
    """
    Generate sample data for name matching.
    
    Parameters:
    n_samples: Number of name pairs to generate
    match_ratio: Ratio of matching pairs to non-matching pairs
    
    Returns:
    DataFrame with columns name1, name2, match
    """
    # Arabic, Indian, Pakistani, and Sri Lankan first names
    first_names = [
        # Arabic names
        "Mohammed", "Ahmed", "Ali", "Omar", "Ibrahim", 
        "Fatima", "Aisha", "Mariam", "Zainab", "Layla",
        # Indian names
        "Raj", "Aarav", "Vikram", "Arjun", "Rohan",
        "Priya", "Anjali", "Deepika", "Neha", "Sunita",
        # Pakistani names
        "Imran", "Faisal", "Tariq", "Kamran", "Asad",
        "Sana", "Ayesha", "Noor", "Hina", "Saima",
        # Sri Lankan names
        "Nuwan", "Chaminda", "Sanath", "Dinesh", "Lasith",
        "Kumari", "Malini", "Shamali", "Amali", "Dilrukshi"
    ]
    
    # Arabic, Indian, Pakistani, and Sri Lankan last names
    last_names = [
        # Arabic surnames
        "Al-Sayed", "Al-Abdullah", "Al-Rahman", "Al-Farsi", "Al-Baghdadi",
        "El-Masri", "Mahmoud", "Hassan", "Khalil", "Amir",
        # Indian surnames
        "Sharma", "Patel", "Singh", "Kumar", "Gupta",
        "Agarwal", "Rao", "Reddy", "Mukherjee", "Joshi",
        # Pakistani surnames
        "Khan", "Ahmed", "Malik", "Qureshi", "Siddiqui",
        "Chaudhry", "Sheikh", "Abbasi", "Butt", "Javed",
        # Sri Lankan surnames
        "Perera", "Fernando", "Silva", "Dissanayake", "Bandara",
        "Jayawardene", "Wickramasinghe", "Senanayake", "Herath", "Mendis"
    ]
    
    # Function to introduce variations with culture-specific variations
    def vary_name(name):
        variations = [
            lambda n: n,  # No change
            lambda n: n.lower(),  # All lowercase
            lambda n: n.upper(),  # All uppercase
            lambda n: n[0] + '.' if len(n) > 0 else '',  # Initial with period
            lambda n: n[0] if len(n) > 0 else '',  # Just initial
            lambda n: n + 's' if not n.endswith('s') else n,  # Add 's'
            lambda n: n.replace('a', 'e') if 'a' in n else n,  # Replace 'a' with 'e'
            lambda n: n.replace('i', 'y') if 'i' in n else n,  # Replace 'i' with 'y'
            lambda n: n + '-' + random.choice(last_names) if random.random() < 0.2 else n,  # Hyphenated name
            lambda n: n[:len(n)//2] if len(n) > 3 else n,  # Truncated
            lambda n: n + ' ' + random.choice(first_names)[0] if random.random() < 0.1 else n,  # Add middle initial
            # Culture-specific variations
            lambda n: 'Al ' + n if random.random() < 0.1 else n,  # Arabic prefix
            lambda n: 'Bin ' + n if random.random() < 0.1 else n,  # Arabic patronymic
            lambda n: 'Abd ' + n if random.random() < 0.1 else n,  # Arabic prefix
            lambda n: n.replace('oo', 'u') if 'oo' in n else n,  # Common transliteration variation
            lambda n: n.replace('ee', 'i') if 'ee' in n else n,  # Common transliteration variation
            lambda n: n.replace('th', 't') if 'th' in n else n,  # Transliteration variation
            lambda n: n.replace('ph', 'f') if 'ph' in n else n,  # Transliteration variation
            lambda n: n.replace('sh', 's') if 'sh' in n else n,  # Transliteration variation
            lambda n: n + 'ji' if random.random() < 0.1 else n,  # Indian honorific suffix
            lambda n: n + 'bhai' if random.random() < 0.1 else n,  # Indian/Pakistani suffix
        ]
        return random.choice(variations)(name)
    
    # Function to introduce typos with culture-specific variations
    def introduce_typos(name, p=0.3):
        if random.random() > p:
            return name
        
        name = list(name)
        ops = ['insert', 'delete', 'replace', 'transpose', 'cultural_variation']
        op = random.choice(ops)
        
        if op == 'insert' and len(name) > 0:
            pos = random.randint(0, len(name))
            # Include characters common in transliteration of Arabic, Indian names
            char = random.choice('abcdefghijklmnopqrstuvwxyz\'-')
            name.insert(pos, char)
        elif op == 'delete' and len(name) > 1:
            pos = random.randint(0, len(name)-1)
            name.pop(pos)
        elif op == 'replace' and len(name) > 0:
            pos = random.randint(0, len(name)-1)
            # Include characters common in transliteration
            char = random.choice('abcdefghijklmnopqrstuvwxyz\'-')
            name[pos] = char
        elif op == 'transpose' and len(name) > 1:
            pos = random.randint(0, len(name)-2)
            name[pos], name[pos+1] = name[pos+1], name[pos]
        elif op == 'cultural_variation' and len(name) > 2:
            # Common transliteration variations
            variations = [
                ('ph', 'f'), ('f', 'ph'),
                ('z', 's'), ('s', 'z'),
                ('ee', 'i'), ('i', 'ee'),
                ('oo', 'u'), ('u', 'oo'),
                ('th', 't'), ('t', 'th'),
                ('v', 'w'), ('w', 'v'),
                ('kh', 'k'), ('k', 'kh'),
                ('a', 'ah'), ('ah', 'a'),
                ('d', 'dh'), ('dh', 'd'),
                ('c', 'k'), ('k', 'c')
            ]
            
            # Join back to string to apply replacements
            name_str = ''.join(name)
            
            # Try to apply a random variation
            for _ in range(3):  # Try up to 3 times
                old_str, new_str = random.choice(variations)
                if old_str in name_str.lower():
                    pos = name_str.lower().find(old_str)
                    name_str = name_str[:pos] + new_str + name_str[pos+len(old_str):]
                    break
                    
            # Convert back to list
            name = list(name_str)
            
        return ''.join(name)
    
    data = []
    
    # Generate matching pairs (with variations)
    n_matches = int(n_samples * match_ratio)
    for _ in range(n_matches):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        # Decide if we should vary first name, last name, or both
        vary_first = random.random() < 0.7
        vary_last = random.random() < 0.7
        
        # Decide if we should swap first/last name order (common in many cultures)
        swap_name_order = include_name_order_swap and random.random() < 0.2
        
        if swap_name_order:
            name1 = f"{first_name} {last_name}"
            name2 = f"{vary_name(last_name) if vary_last else last_name}, {vary_name(first_name) if vary_first else first_name}"
        else:
            name1 = f"{first_name} {last_name}"
            name2 = f"{vary_name(first_name) if vary_first else first_name} {vary_name(last_name) if vary_last else last_name}"
        
        # Add middle names or initials (common in South Asian names)
        if random.random() < 0.15:
            middle = random.choice(first_names)
            if random.random() < 0.5:
                middle = middle[0] + "."
            name2 = f"{name2.split(' ')[0]} {middle} {' '.join(name2.split(' ')[1:])}"
            
        # Introduce typos
        name2 = introduce_typos(name2)
        
        data.append({
            'name1': name1,
            'name2': name2,
            'match': 1
        })
    
    # Generate non-matching pairs
    n_nonmatches = n_samples - n_matches
    for _ in range(n_nonmatches):
        first_name1 = random.choice(first_names)
        last_name1 = random.choice(last_names)
        
        # Choose different names for non-matches
        first_name2 = random.choice([n for n in first_names if n != first_name1])
        last_name2 = random.choice([n for n in last_names if n != last_name1])
        
        # Sometimes keep one part the same
        if random.random() < 0.3:
            if random.random() < 0.5:
                first_name2 = first_name1
            else:
                last_name2 = last_name1
        
        name1 = f"{first_name1} {last_name1}"
        name2 = f"{first_name2} {last_name2}"
        
        # Introduce typos
        name2 = introduce_typos(name2)
        
        data.append({
            'name1': name1,
            'name2': name2,
            'match': 0
        })
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)

# Generate sample data with cultural variations
train_data = generate_sample_data(800, 0.5, include_name_order_swap=True)
test_data = generate_sample_data(200, 0.5, include_name_order_swap=True)

# Save to CSV
train_data.to_csv('name_matching_train.csv', index=False)
test_data.to_csv('name_matching_test.csv', index=False)

print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")
print("\nSample of training data:")
print(train_data.head())