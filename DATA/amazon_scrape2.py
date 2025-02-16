import os
import pandas as pd

data = {
    'Options': [
        'Without Cable, Retractable, Detachable, Tangle Free',
        'Bud, Hook, Oval, Circle, Rectangle, Square, Stick',
        'Not Water Resistant, Water-Resistant, Waterproof',
        'UrbanUSA, MoraLana',
        'Media Control, Call Control, Volume Control, Alexa, App Control, Button Control, Google Assistant, Noise Control, Siri, Touch Control, Voice Control',
        'In Ear, On Ear, Over Ear, True Wireless',
        'Last 30 Days, Last 90 Days',
        '3 in & above',
        'Up to 19 ms, 20 to 34 ms, 35 to 49 ms, 50 ms & above',
        'Climate Pledge Friendly',
        'Up to 99,999 Hz, 100,000 to 199,999 Hz, 200,000 to 299,999 Hz',
        'Plastic, Fabric, Leather, Aluminum, Cardboard, Faux Leather, High-Density Polyethylene (HDPE)',
        'Up to 1 h, 1 to 1.9 h, 2 to 2.9 h, 3 h & above',
        'Black, White, Pink, Beige, Blue, Brown, Clear',
        'Travel, Gaming, Entertainment, Business, Fitness, Professional, School, Sleeping',
        'Dynamic Driver, Bone Conduction Driver, Hybrid Driver, Balanced Armature Driver, Electrostatic Driver, Piezoelectric Driver, Planar Magnetic Driver',
        '1, 2, 3 & above',
        'Up to 9 h, 10 to 14 h, 15 to 19 h, 20 to 24 h, 25 h & above',
        'Up to 1, 1 to 1.9, 2 to 2.9, 3 & above',
        'Up to 1 lb, 1 to 1.9 lb, 2 to 2.9 lb, 3 lb & above'
    ]
}

# Split each option into separate cells
split_data = [option.split(', ') for option in data['Options']]

# Create a new DataFrame
df = pd.DataFrame(split_data)

# Save the DataFrame to a CSV file
df.to_csv('headphones_filters.csv', index=False)
print("Data saved to headphones_filters.csv")