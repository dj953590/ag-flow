# main.py
"""
Main application entry point
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agent.mcp_agent import MCPEnhancedCDEExtractor

async def main():
    """Main application"""

    print("\n" + "="*60)
    print("🔍 CDE Extraction System with MCP Tools")
    print("="*60 + "\n")

    # Initialize agent
    agent = MCPEnhancedCDEExtractor("config/mcp_config.yaml")

    try:
        # Initialize agent and tools
        await agent.initialize()

        # Example CDE prompts
        cde_prompts = [
            {
                "user_query": "from given set of documents identify distinct facility and maturity date for each facility",
                "instructions": "Carefully scan the documents for all the facility mentioned in the documents and identify the maturity date. Maturity date may mention Closing date which can be derived from the first page of the document.",
                "step_by_step_reasoning": "1. First identify each distinct facility from the credit agreement using the provided tools 2. Identify the closing date of the agreement 3) for Each facility identity mention of the maturity date 4. if maturity date mentions the closing date then calculate the maturity date from the closing date",
                "output": {
                    "facility": "Term A Loan",
                    "maturity_date": "09-21-2027",
                    "justification": "Document mentions",
                    "confidence_score": 0.95
                }
            },
            {
                "user_query": "extract all interest rate provisions and their applicable facilities",
                "instructions": "Find all interest rate mentions and map them to specific facilities",
                "step_by_step_reasoning": "1. Search for interest rate keywords 2. Extract rate provisions 3. Identify which facility each rate applies to 4. Validate rates are consistent",
                "output": {
                    "facility": "Term A Loan",
                    "interest_rate": "LIBOR + 2.50%",
                    "effective_date": "Closing Date"
                }
            }
        ]

        # Process each CDE
        for i, cde_prompt in enumerate(cde_prompts):
            print(f"\n📄 Processing CDE {i+1}: {cde_prompt['user_query'][:50]}...")

            result = await agent.extract_cde(
                cde_prompt=cde_prompt,
                document_id="credit_agreement_001"
            )

            # Display results
            print(f"\n✅ Extraction Complete!")
            print(f"   Overall Confidence: {result.overall_confidence:.2f}")
            print(f"   Validation Status: {result.validation_status}")

            if result.extracted_data.get("facilities"):
                print(f"\n📊 Extracted Facilities:")
                for facility in result.extracted_data["facilities"]:
                    print(f"   • {facility['facility']} (confidence: {facility['extraction_confidence']:.2f})")

            # Save detailed results
            output_file = f"results/cde_result_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("results", exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(result.dict(), f, indent=2, default=str)

            print(f"\n💾 Results saved to: {output_file}")

        print("\n" + "="*60)
        print("🎉 All CDE extractions completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean shutdown
        await agent.shutdown()

if __name__ == "__main__":
    # Set OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Run async main
    asyncio.run(main())