#!/usr/bin/env python3
"""
Test script for the Safety AI functionality
"""

import requests
import json
import time

def test_safety_ai_api():
    """Test the Safety AI API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🛡️ Testing Safety AI for FAST Students")
    print("=" * 50)
    
    # Test 1: Welcome endpoint
    print("\n1. Testing Safety AI Welcome...")
    try:
        response = requests.get(f"{base_url}/api/safety-ai/welcome")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Welcome loaded: {data['title']}")
            print(f"   Features: {len(data['features'])} available")
            print(f"   Quick tips: {len(data['quick_tips'])} tips")
        else:
            print(f"❌ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 2: Travel Guide
    print("\n2. Testing Travel Guide...")
    try:
        response = requests.get(f"{base_url}/api/safety-ai/travel-guide")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Travel Guide: {data['title']}")
            sections = list(data['sections'].keys())
            print(f"   Sections: {', '.join(sections)}")
            for section, content in data['sections'].items():
                print(f"   - {content['title']}: {len(content['tips'])} tips")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Emergency Preparation
    print("\n3. Testing Emergency Preparation...")
    try:
        response = requests.get(f"{base_url}/api/safety-ai/emergency-prep")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Emergency Guide: {data['title']}")
            print(f"   Emergency contacts: {len(data['emergency_contacts'])}")
            print(f"   Preparation checklist: {len(data['preparation_checklist'])} items")
            print(f"   Weather emergencies: {len(data['weather_emergencies']['tips'])} tips")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Campus Safety
    print("\n4. Testing Campus Safety...")
    try:
        response = requests.get(f"{base_url}/api/safety-ai/campus-safety")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Campus Safety: {data['title']}")
            sections = list(data['sections'].keys())
            print(f"   Sections: {', '.join(sections)}")
            for section, content in data['sections'].items():
                print(f"   - {content['title']}: {len(content['tips'])} tips")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Weather Safety
    print("\n5. Testing Weather Safety...")
    try:
        response = requests.get(f"{base_url}/api/safety-ai/weather-safety")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Weather Safety: {data['title']}")
            weather = data['current_weather']
            print(f"   Current weather: {weather['temperature']}°C, {weather['description']}")
            print(f"   Recommendations: {len(data['recommendations'])}")
            print(f"   Safety tip categories: {len(data['safety_tips'])}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 6: Personal Checklist
    print("\n6. Testing Personal Checklist...")
    try:
        response = requests.get(f"{base_url}/api/safety-ai/personal-checklist")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Personal Checklist: {data['title']}")
            print(f"   Daily checklist: {len(data['daily_checklist'])} items")
            print(f"   Weekly checklist: {len(data['weekly_checklist'])} items")
            print(f"   Monthly checklist: {len(data['monthly_checklist'])} items")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Safety AI testing completed!")
    print("\n🎯 Safety AI Features Available:")
    print("   • Welcome page with animated avatar")
    print("   • Travel safety guide for FAST students")
    print("   • Emergency contacts and preparation")
    print("   • Campus-specific safety tips")
    print("   • Weather-based safety recommendations")
    print("   • Personal safety checklists")
    print("\n🚀 Access the Safety AI by clicking the 'Safety AI' button on the main page!")
    
    return True

if __name__ == "__main__":
    print("🛡️ Starting Safety AI test...")
    print("Make sure the Flask app is running on http://localhost:5000")
    print("Press Ctrl+C to stop the test")
    
    try:
        test_safety_ai_api()
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
