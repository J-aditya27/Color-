# skincare_recommendations.py

skincare_suggestions = {
    "redness": {
        "mild": {
            "products": [
                {"name": "Cetaphil Gentle Skin Cleanser", "type": "Cleanser"},
                {"name": "CeraVe Moisturizing Cream", "type": "Moisturizer"},
                {"name": "La Roche-Posay Anthelios Mineral Sunscreen", "type": "Sunscreen"}
            ],
            "medications": [
                {"name": "Hydrocortisone 1% Cream", "usage": "Apply thinly to red areas to reduce inflammation."}
            ]
        },
        "severe": {
            "products": [
                {"name": "The Ordinary Niacinamide 10% + Zinc 1%", "type": "Serum"},
                {"name": "Paula’s Choice Azelaic Acid Booster", "type": "Treatment"}
            ],
            "medications": [
                {"name": "Metronidazole Gel (0.75%)", "usage": "Apply as prescribed for rosacea or persistent redness."}
            ]
        }
    },
    "dryness": {
        "mild": {
            "products": [
                {"name": "Neutrogena Hydro Boost Cleanser", "type": "Cleanser"},
                {"name": "The Ordinary Hyaluronic Acid 2% + B5", "type": "Serum"}
            ],
            "medications": [
                {"name": "Urea 10% Cream", "usage": "Apply to severely dry patches to restore hydration."}
            ]
        },
        "severe": {
            "products": [
                {"name": "Aquaphor Healing Ointment", "type": "Ointment"},
                {"name": "Laneige Water Sleeping Mask", "type": "Overnight Mask"}
            ],
            "medications": [
                {"name": "Ammonium Lactate Cream", "usage": "Prescribed for extreme dryness or scaling."}
            ]
        }
    },
    "pimples": {
        "mild": {
            "products": [
                {"name": "Clean & Clear Advantage Spot Treatment", "type": "Spot Treatment"},
                {"name": "Body Shop Tea Tree Mattifying Lotion", "type": "Moisturizer"}
            ],
            "medications": [
                {"name": "Clindamycin Gel", "usage": "Apply to pimples twice daily."}
            ]
        },
        "moderate": {
            "products": [
                {"name": "PanOxyl 4% Creamy Wash", "type": "Cleanser"},
                {"name": "Differin Gel", "type": "Retinol Treatment"}
            ],
            "medications": [
                {"name": "Benzoyl Peroxide 5% Gel", "usage": "Use as a spot treatment or prescribed by a dermatologist."},
                {"name": "Adapalene Gel", "usage": "Apply nightly for acne treatment."}
            ]
        }
    }
}
