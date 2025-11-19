import httpx
import asyncio
import sys

async def test_chat():
    try:
        async with httpx.AsyncClient() as client:
            print("🚀 Testing MedIntel Backend API...")
            print("=" * 60)
            
            response = await client.post(
                "http://localhost:8000/api/v1/chat",
                json={
                    "question": "What are the symptoms of diabetes?",
                    "context": "",
                    "model_provider": "gemini",
                    "student_mode": False,
                    "mode": "medical"
                },
                timeout=30.0
            )
            
            print(f"\n✅ Status: {response.status_code}")
            data = response.json()
            
            print(f"\n📝 Summary:\n{data.get('summary', 'N/A')}")
            print(f"\n💬 Answer:\n{data.get('answer', 'N/A')[:300]}...")
            print(f"\n⚠️  Risk Level: {data.get('risk_level', 'N/A')}")
            print(f"🎯 Confidence: {data.get('confidence', 'N/A')}")
            print(f"😊 Emotion: {data.get('emotion', 'N/A')}")
            print(f"\n📚 Citations: {', '.join(data.get('citations', []))}")
            print(f"\n📋 Next Steps:")
            for step in data.get('next_steps', []):
                print(f"   • {step}")
            
            if "Demo Mode Active" in data.get('answer', ''):
                print("\n" + "=" * 60)
                print("❌ DEMO MODE - API key not working")
                print("=" * 60)
                sys.exit(1)
            else:
                print("\n" + "=" * 60)
                print("✅ SUCCESS! Gemini 2.0-flash is responding!")
                print("=" * 60)
                sys.exit(0)
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_chat())
