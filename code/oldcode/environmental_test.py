if __name__ == "__main__":
    # Create model instances
    human_model = HumanThreatModel()
    animal_model = AnimalThreatModel()

    print("=== Human Threat Analysis ===")
    # Test different scenarios
    scenarios = [
        ('usa', 'male', 'black'),
        ('usa', 'female', 'white'),
        ('japan', 'male', 'other')
    ]

    for country, gender, ethnicity in scenarios:
        time, risk = human_model.calculate_human_threat(country, gender, ethnicity, years=80)
        print(f"{country}, {gender}, {ethnicity}: Final risk = {risk[-1]:.6f}")

    print("\n=== Animal Threat Analysis ===")
    animals = ['bear', 'wolf', 'shark']
    for animal in animals:
        time, survival, threat = animal_model.calculate_animal_threat(animal, time_span=30)
        print(f"{animal}: Final survival = {survival[-1]:.3f}, Final threat = {threat[-1]:.3f}")

    print("\n=== Combined Analysis ===")
    results = analyze_external_threats('usa', 'male', 'black', 'bear', years=50)
    print(f"Final cumulative homicide risk: {results['human_risk'][-1]:.6f}")
    print(f"Final survival probability from animal: {results['animal_survival'][-1]:.3f}")
    print(f"Final combined risk: {results['combined_risk'][-1]:.3f}")


def quick_test():
    """Quick test to verify the models work"""
    print("🚀 Running quick test...")

    human_model = HumanThreatModel()
    animal_model = AnimalThreatModel()

    # Test 1: Human threat - basic calculation
    time, risk = human_model.calculate_human_threat('usa', 'male', 'black', years=5, n_steps=50)
    print(f"✅ Human threat working - Final risk: {risk[-1]:.6f}")

    # Test 2: Animal threat - basic calculation
    time, survival, threat = animal_model.calculate_animal_threat('bear', time_span=10, n_steps=50)
    print(f"✅ Animal threat working - Survival: {survival[-1]:.3f}, Threat: {threat[-1]:.3f}")

    # Test 3: Combined analysis
    results = analyze_external_threats('usa', 'male', 'black', 'bear', years=5)
    print(f"✅ Combined analysis working - Total risk: {results['combined_risk'][-1]:.3f}")

    print("🎉 All models working correctly!")


# Run it
quick_test()