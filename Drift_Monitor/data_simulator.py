import pandas as pd
import numpy as np

def get_reference_data(n=1000):
    """
    Generates 'Normal' PPE Detection Data.
    """
    np.random.seed(42)
    
    # 1. Create the Numeric Data first
    data = pd.DataFrame({
        'Helmet_Conf': np.random.normal(0.92, 0.05, n),
        'Vest_Conf': np.random.normal(0.88, 0.06, n),
        'Harness_Conf': np.random.normal(0.90, 0.04, n),
    })
    
    # 2. Clip the numbers (Keep them between 0 and 1)
    # This works now because 'Camera_Zone' isn't here yet.
    data = data.clip(0.0, 1.0)

    # 3. Add the Text/Metadata Column AFTER clipping
    data['Camera_Zone'] = np.random.choice(['Zone_Entry', 'Zone_Mining', 'Zone_Heights'], n)
    
    return data

def get_drifted_data(n=1000, severity='medium'):
    """
    Generates 'Corrupted' PPE Data.
    """
    np.random.seed(99)
    
    # 1. Create Numeric Data
    data = pd.DataFrame({
        'Helmet_Conf': np.random.normal(0.92, 0.05, n),
        'Vest_Conf': np.random.normal(0.88, 0.06, n),
        'Harness_Conf': np.random.normal(0.90, 0.04, n),
    })

    # 2. Apply Drift Logic
    if severity == 'medium':
        data['Vest_Conf'] = data['Vest_Conf'] - 0.20 
        data['Helmet_Conf'] = data['Helmet_Conf'] - 0.10
        
    elif severity == 'high':
        data['Helmet_Conf'] = data['Helmet_Conf'] * 0.5 
        data['Harness_Conf'] = data['Harness_Conf'] * 0.4
        data['Vest_Conf'] = data['Vest_Conf'] * 0.6
    
    # 3. Clip the numbers
    data = data.clip(0.0, 1.0)
    
    # 4. Add the Text Column last
    data['Camera_Zone'] = np.random.choice(['Zone_Entry', 'Zone_Mining', 'Zone_Heights'], n)
    
    return data
