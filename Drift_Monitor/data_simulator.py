import pandas as pd
import numpy as np

def get_reference_data(n=1000):
    """
    Generates 'Normal' PPE Detection Data.
    Simulates a well-lit environment where workers are compliant 
    and the model detects gear with HIGH CONFIDENCE.
    """
    np.random.seed(42)
    data = pd.DataFrame({
        # Model is very sure (85-99%) about Helmets
        'Helmet_Conf': np.random.normal(0.92, 0.05, n),
        
        # Model is sure about Vests
        'Vest_Conf': np.random.normal(0.88, 0.06, n),
        
        # Safety Harness (Critical for heights)
        'Harness_Conf': np.random.normal(0.90, 0.04, n),
        
        # Metadata
        'Camera_Zone': np.random.choice(['Zone_Entry', 'Zone_Mining', 'Zone_Heights'], n)
    })
    # Clip to realistic probabilities (0 to 1)
    return data.clip(0.0, 1.0)

def get_drifted_data(n=1000, severity='medium'):
    """
    Generates 'Corrupted' PPE Data.
    Simulates Fog/Dust or Non-Compliance where model confidence DROPS.
    """
    np.random.seed(99)
    data = pd.DataFrame({
        'Helmet_Conf': np.random.normal(0.92, 0.05, n),
        'Vest_Conf': np.random.normal(0.88, 0.06, n),
        'Harness_Conf': np.random.normal(0.90, 0.04, n),
        'Camera_Zone': np.random.choice(['Zone_Entry', 'Zone_Mining', 'Zone_Heights'], n)
    })

    if severity == 'medium':
        # Scenario: Dusty Lens / Dirty Vests
        # Model struggles to see Vests (Confidence drops)
        data['Vest_Conf'] = data['Vest_Conf'] - 0.20 
        data['Helmet_Conf'] = data['Helmet_Conf'] - 0.10
        
    elif severity == 'high':
        # Scenario: Heavy Fog or New Helmet Color (Model failure)
        # Massive drop in confidence. System should LOCK DOWN.
        data['Helmet_Conf'] = data['Helmet_Conf'] * 0.5  # Model is blind
        data['Harness_Conf'] = data['Harness_Conf'] * 0.4 # Danger!
        data['Vest_Conf'] = data['Vest_Conf'] * 0.6
    
    return data.clip(0.0, 1.0)