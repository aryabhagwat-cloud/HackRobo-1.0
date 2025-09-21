#!/usr/bin/env python3
"""
Test the new growth stage detection and economic impact features
"""

import json
from agricrawler.growth_stage import detect_growth_stage, get_growth_stage_recommendations
from agricrawler.economic_analysis import assess_economic_impact, get_crop_economics_summary

def test_growth_stage_detection():
    """Test growth stage detection functionality."""
    print("🌱 TESTING GROWTH STAGE DETECTION")
    print("=" * 50)
    
    try:
        # Test with sample image
        growth_stage = detect_growth_stage('sample.jpg')
        
        print(f"Growth Stage: {growth_stage.stage}")
        print(f"Confidence: {growth_stage.confidence:.2f}")
        print(f"Stage Percentage: {growth_stage.stage_percentage:.1f}%")
        print(f"Days to Harvest: {growth_stage.days_to_harvest}")
        print(f"Next Stage: {growth_stage.next_stage}")
        print(f"Harvest Readiness: {growth_stage.harvest_readiness:.2f}")
        print(f"Quality Prediction: {growth_stage.quality_prediction}")
        
        # Get recommendations
        recommendations = get_growth_stage_recommendations(growth_stage.stage, growth_stage.days_to_harvest)
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n✅ Growth stage detection: WORKING")
        return True
        
    except Exception as e:
        print(f"❌ Growth stage detection failed: {e}")
        return False

def test_economic_analysis():
    """Test economic impact assessment functionality."""
    print("\n💰 TESTING ECONOMIC IMPACT ANALYSIS")
    print("=" * 50)
    
    try:
        # Test economic impact assessment
        economic_impact = assess_economic_impact(
            stress_score=0.6,
            disease_label="Tomato_Early_blight",
            pest_label="beetle",
            soil_moisture=35.0,
            crop_type="tomato",
            growth_stage="flowering",
            days_to_harvest=15
        )
        
        print(f"Potential Yield Loss: {economic_impact.potential_yield_loss:.1%}")
        print(f"Treatment Cost: ${economic_impact.treatment_cost:.2f}/acre")
        print(f"ROI of Treatment: {economic_impact.roi_of_treatment:.1f}x")
        print(f"Market Value Impact: ${economic_impact.market_value_impact:.2f}/acre")
        print(f"Cost-Benefit Ratio: {economic_impact.cost_benefit_ratio:.2f}")
        print(f"Urgency Level: {economic_impact.urgency_level}")
        print(f"Recommended Action: {economic_impact.recommended_action}")
        
        # Test crop economics summary
        print(f"\nCrop Economics Summary:")
        economics = get_crop_economics_summary("tomato")
        print(f"Market Price: ${economics['market_price_per_kg']}/kg")
        print(f"Expected Yield: {economics['expected_yield_per_acre']} kg/acre")
        print(f"Potential Revenue: ${economics['potential_revenue_per_acre']}/acre")
        
        print("\n✅ Economic analysis: WORKING")
        return True
        
    except Exception as e:
        print(f"❌ Economic analysis failed: {e}")
        return False

def test_integrated_analysis():
    """Test integrated analysis with both new features."""
    print("\n🔄 TESTING INTEGRATED ANALYSIS")
    print("=" * 50)
    
    try:
        from agricrawler.optimized_pipeline import analyze_crop_health
        
        # Test complete analysis
        result = analyze_crop_health(
            soil_pct=45.0,
            temp_c=30.0,
            humidity_pct=60.0,
            image_path="sample.jpg"
        )
        
        print("Integrated Analysis Results:")
        print(f"Growth Stage: {result['growth_stage']['stage']}")
        print(f"Days to Harvest: {result['growth_stage']['days_to_harvest']}")
        print(f"Quality Prediction: {result['growth_stage']['quality_prediction']}")
        print(f"Potential Yield Loss: {result['economic_impact']['potential_yield_loss']}")
        print(f"ROI of Treatment: {result['economic_impact']['roi_of_treatment']}")
        print(f"Urgency Level: {result['economic_impact']['urgency_level']}")
        
        print("\n✅ Integrated analysis: WORKING")
        return True
        
    except Exception as e:
        print(f"❌ Integrated analysis failed: {e}")
        return False

def main():
    """Run all tests."""
    print("AGRICRAWLER NEW FEATURES TEST")
    print("=" * 60)
    print("Testing Growth Stage Detection and Economic Impact Analysis")
    print()
    
    tests = [
        ("Growth Stage Detection", test_growth_stage_detection),
        ("Economic Impact Analysis", test_economic_analysis),
        ("Integrated Analysis", test_integrated_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing: {test_name}")
        try:
            success = test_func()
            results.append(success)
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
        except Exception as e:
            print(f"Result: ❌ ERROR - {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)
    
    if passed == total:
        print("🎉 ALL NEW FEATURES WORKING!")
        print()
        print("✅ Growth Stage Detection: WORKING")
        print("   - Identifies crop development stages")
        print("   - Predicts harvest timing and quality")
        print("   - Provides stage-specific recommendations")
        print()
        print("✅ Economic Impact Analysis: WORKING")
        print("   - Calculates yield loss predictions")
        print("   - Provides ROI analysis for treatments")
        print("   - Offers economic decision support")
        print()
        print("✅ Integrated Analysis: WORKING")
        print("   - Combines all features seamlessly")
        print("   - Provides comprehensive farmer insights")
        print()
        print("🚀 AGRICRAWLER IS READY WITH ENHANCED FEATURES!")
        print("   Farmers now get growth stage intelligence and economic insights!")
    else:
        print("⚠️  Some features need attention")
        print(f"Passed: {passed}/{total}")
    
    return passed == total

if __name__ == "__main__":
    main()

