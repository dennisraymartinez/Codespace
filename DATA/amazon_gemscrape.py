import csv

# Data
data = [
    {
        "product_name": "soundcore Life Q30 by Anker, Hybrid Active Noise Cancelling Headphones, with Multiple Modes, Hi-Res Sound, Custom EQ via App, 50H Playtime, Comfortable Fit, Bluetooth, Multipoint Connection",
        "number_of_reviews": 75442
    },
    {
        "product_name": "NANK Runner Diver 2 Pro - Bone Conduction Headphones with Noise Cancelling Mode, IP69 Swimming Headphones, Bluetooth 5.4 & 32GB MP3 Player, Open Ear Headphones with Mic, Fit for Sports",
        "number_of_reviews": 45
    },
    {
        "product_name": "Wired Over Ear Headphones, Studio Monitor & Mixing DJ Headphones with 50mm Neodymium Drivers and 1/4 to 3.5mm Jack for Guitar AMP Podcast Piano Keyboard (Black)",
        "number_of_reviews": 833
    },
    {
        "product_name": "Beats Studio Pro x Kim Kardashian - Bluetooth Noise Cancelling Headphones,Personalized Spatial Audio, USB-C Lossless Audio, Apple & Android Compatibility, Up to 40 Hours Battery Life - Dune",
        "number_of_reviews": 14259
    },
    {
        "product_name": "Amazon Basics Bluetooth Headphones with Microphone, Wireless, On Ear, 35 Hour Playtime, Foldable, One Size, Black",
        "number_of_reviews": 534
    },
    {
        "product_name": "Soundcore Anker Life Q20 Hybrid Active Noise Cancelling Headphones, Wireless Over Ear Bluetooth Headphones, 60H Playtime, Hi-Res Audio, Deep Bass, Memory Foam Ear Cups, for Travel, Home Office",
        "number_of_reviews": 83144
    },
    {
        "product_name": "Bose QuietComfort Bluetooth Headphones, Wireless Headphones, Over Ear Noise Cancelling Headphones with Mic, Up To 24 Hours of Battery Life, Black",
        "number_of_reviews": 7400
    },
    {
        "product_name": "Sony WF-C510 Truly Wireless in-Ear Bluetooth Earbud Headphones with up to 22-Hour Battery, Multipoint Connection, Mic and IPX4 Water Resistance, Yellow- New",
        "number_of_reviews": 105
    },
    {
        "product_name": "Soundcore by Anker, Space One, Active Noise Cancelling Headphones, 2X Stronger Voice Reduction, 40H ANC Playtime, App Control, LDAC Hi-Res Wireless Audio, Comfortable Fit, Clear Calls, Bluetooth 5.3",
        "number_of_reviews": 6595
    },
    {
        "product_name": "JBL Tune 520BT - Wireless On-Ear Headphones, Up to 57H Battery Life and Speed Charge, Lightweight, Comfortable and Foldable Design, Hands-Free Calls with Voice Aware (Purple)",
        "number_of_reviews": 2730
    },
    {
        "product_name": "Sony WH-1000XM4 Wireless Premium Noise Canceling Overhead Headphones with Mic for Phone-Call and Alexa Voice Control, Silver WH1000XM4",
        "number_of_reviews": 58566
    },
    {
        "product_name": "JLab JBuds Lux ANC Wireless Headphones, Graphite, Hybrid Active Noise Cancelling, Customizable Sound, Spatial Audio Compatible, Premium Over-Ear Bluetooth Headset",
        "number_of_reviews": 854
    },
    {
        "product_name": "Beats Studio Buds - True Wireless Noise Cancelling Earbuds - Compatible with Apple & Android, Built-in Microphone, IPX4 Rating, Sweat Resistant Earphones, Class 1 Bluetooth Headphones - Red",
        "number_of_reviews": 80270
    }
]

# Filepath to save the CSV file
csv_file_path = 'd:/Codespace/amazon_gemscrape.csv'

# Writing to CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["product_name", "number_of_reviews"])
    writer.writeheader()
    writer.writerows(data)

print(f"CSV file has been created at {csv_file_path}")