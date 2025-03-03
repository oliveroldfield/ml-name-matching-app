import pandas as pd
import random
import numpy as np

# Generate sample training data
def generate_sample_data(n_samples=1000, match_ratio=0.5):
    """
    Generate sample data for name matching.
    
    Parameters:
    n_samples: Number of name pairs to generate
    match_ratio: Ratio of matching pairs to non-matching pairs
    
    Returns:
    DataFrame with columns name1, name2, match
    """
    # List of common first names
    first_names = [
        "James", "John", "Robert", "Michael", "William", 
        "David", "Joseph", "Charles", "Thomas", "Daniel",
        "Mary", "Patricia", "Jennifer", "Elizabeth", "Linda",
        "Barbara", "Susan", "Jessica", "Margaret", "Sarah"
    ]
    
    # List of common last names
    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown",
        "Davis", "Miller", "Wilson", "Moore", "Taylor",
        "Anderson", "Thomas", "Jackson", "White", "Harris",
        "Martin", "Thompson", "Garcia", "Martinez", "Robinson"
    ]
    
    # Function to introduce variations
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
        ]
        return random.choice(variations)(name)
    
    # Function to introduce typos
    def introduce_typos(name, p=0.3):
        if random.random() > p:
            return name
        
        name = list(name)
        ops = ['insert', 'delete', 'replace', 'transpose']
        op = random.choice(ops)
        
        if op == 'insert' and len(name) > 0:
            pos = random.randint(0, len(name))
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            name.insert(pos, char)
        elif op == 'delete' and len(name) > 1:
            pos = random.randint(0, len(name)-1)
            name.pop(pos)
        elif op == 'replace' and len(name) > 0:
            pos = random.randint(0, len(name)-1)
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            name[pos] = char
        elif op == 'transpose' and len(name) > 1:
            pos = random.randint(0, len(name)-2)
            name[pos], name[pos+1] = name[pos+1], name[pos]
            
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
        
        name1 = f"{first_name} {last_name}"
        name2 = f"{vary_name(first_name) if vary_first else first_name} {vary_name(last_name) if vary_last else last_name}"
        
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

# Generate sample data
train_data = generate_sample_data(800, 0.5)
test_data = generate_sample_data(200, 0.5)

# Save to CSV
train_data.to_csv('name_matching_train.csv', index=False)
test_data.to_csv('name_matching_test.csv', index=False)

print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples")
print("\nSample of training data:")
print(train_data.head())