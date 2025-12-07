# mcp_servers/document_tools.py
"""
MCP Server for document retrieval tools
"""
import json
import re
from typing import Dict, List, Any, Optional
from mcp import FastMCP, StdioServerParameters
from datetime import datetime, timedelta
import asyncio

# Initialize MCP server
mcp = FastMCP("document_retrieval_tools")

# Simulated document store
DOCUMENTS = {
    "credit_agreement_001": {
        "title": "Credit Agreement between ABC Corp and XYZ Bank",
        "closing_date": "2023-09-21",
        "pages": {
            1: """CREDIT AGREEMENT
                
                Dated as of September 21, 2023 (the "Closing Date")
                
                BETWEEN:
                
                ABC CORPORATION, as Borrower
                
                AND
                
                XYZ BANK, as Lender
                
                ARTICLE I
                DEFINITIONS
                
                1.01 Defined Terms. When used in this Agreement, the following terms shall have the meanings specified:
                
                "Term A Loan" means the term loan in the principal amount of $50,000,000.
                "Term B Loan" means the term loan in the principal amount of $30,000,000.
                "Revolving Facility" means the revolving credit facility in an aggregate principal amount of $20,000,000.
                
                """,
            2: """ARTICLE II
                THE CREDIT FACILITIES
                
                2.01 Term A Loan. The Term A Loan shall mature on September 21, 2027.
                
                2.02 Term B Loan. The Term B Loan shall mature on December 2, 2025.
                
                2.03 Revolving Facility. The Revolving Facility shall mature on June 15, 2026.
                
                2.04 Interest Rates. (a) The Term A Loan shall bear interest at LIBOR + 2.50%.
                (b) The Term B Loan shall bear interest at LIBOR + 3.25%.
                (c) The Revolving Facility shall bear interest at LIBOR + 2.00%.
                """,
            3: """ARTICLE III
                REPRESENTATIONS AND WARRANTIES
                
                3.01 Organization and Qualification. The Borrower is duly organized...
                
                ARTICLE IV
                CONDITIONS PRECEDENT
                
                4.01 Conditions to Initial Extension of Credit. The obligation of the Lender...
                """
        }
    }
}

@mcp.tool()
async def retrieve_first_page(document_id: str) -> str:
    """
    Retrieve the first page of a credit agreement document.

    Args:
        document_id: Unique identifier for the document

    Returns:
        First page content with key metadata
    """
    if document_id not in DOCUMENTS:
        return json.dumps({"error": f"Document {document_id} not found"})

    doc = DOCUMENTS[document_id]
    page_content = doc["pages"][1]

    return json.dumps({
        "document_id": document_id,
        "page_number": 1,
        "content": page_content,
        "metadata": {
            "title": doc["title"],
            "closing_date": doc.get("closing_date"),
            "extracted_closing_date": extract_closing_date(page_content)
        },
        "timestamp": datetime.now().isoformat()
    })

@mcp.tool()
async def search_by_keywords(
        document_id: str,
        keywords: List[str],
        context_size: int = 200
) -> str:
    """
    Search document for keywords and return relevant chunks.

    Args:
        document_id: Document to search
        keywords: List of keywords to search for
        context_size: Number of characters around keyword to include

    Returns:
        JSON with matching chunks and their context
    """
    if document_id not in DOCUMENTS:
        return json.dumps({"error": f"Document {document_id} not found"})

    results = []
    doc = DOCUMENTS[document_id]

    for page_num, content in doc["pages"].items():
        content_lower = content.lower()

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Find all occurrences
            start = 0
            while True:
                idx = content_lower.find(keyword_lower, start)
                if idx == -1:
                    break

                # Extract context around keyword
                context_start = max(0, idx - context_size)
                context_end = min(len(content), idx + len(keyword) + context_size)
                context = content[context_start:context_end]

                # Highlight keyword
                highlighted = context.replace(
                    content[idx:idx + len(keyword)],
                    f"**{content[idx:idx + len(keyword)]}**"
                )

                results.append({
                    "page": page_num,
                    "keyword": keyword,
                    "context": highlighted,
                    "position": idx,
                    "full_page": page_num
                })

                start = idx + 1

    return json.dumps({
        "document_id": document_id,
        "keywords_searched": keywords,
        "matches_found": len(results),
        "results": results
    })

@mcp.tool()
async def extract_entities(
        document_id: str,
        entity_types: List[str]
) -> str:
    """
    Extract specific entity types from document.

    Args:
        document_id: Document to analyze
        entity_types: Types of entities to extract (facility, date, amount, party)

    Returns:
        Extracted entities organized by type
    """
    if document_id not in DOCUMENTS:
        return json.dumps({"error": f"Document {document_id} not found"})

    entities = {entity_type: [] for entity_type in entity_types}
    doc = DOCUMENTS[document_id]

    # Simple entity extraction patterns
    patterns = {
        "facility": [
            r'"([^"]+ Loan)"',
            r'([A-Z][a-z]+ Facility)',
            r'(Term [A-B] Loan)',
            r'(Revolving Facility)'
        ],
        "date": [
            r'(\w+ \d{1,2}, \d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(mature[sd]? on (\w+ \d{1,2}, \d{4}))'
        ],
        "amount": [
            r'(\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:\.\d+)? million)',
            r'(\d+(?:\.\d+)? billion)'
        ],
        "party": [
            r'([A-Z][A-Za-z\s]+ (?:Corporation|Corp|Inc|LLC|Bank))',
            r'as (Borrower|Lender|Agent)',
            r'between ([^,]+) and ([^,]+)'
        ]
    }

    for page_num, content in doc["pages"].items():
        for entity_type in entity_types:
            if entity_type in patterns:
                for pattern in patterns[entity_type]:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0]
                        entities[entity_type].append({
                            "entity": match.strip(),
                            "page": page_num,
                            "context": extract_context(content, match)
                        })

    # Deduplicate
    for entity_type in entities:
        seen = set()
        unique_entities = []
        for entity in entities[entity_type]:
            if entity["entity"] not in seen:
                seen.add(entity["entity"])
                unique_entities.append(entity)
        entities[entity_type] = unique_entities

    return json.dumps({
        "document_id": document_id,
        "entity_types_requested": entity_types,
        "entities_found": {k: len(v) for k, v in entities.items()},
        "extracted_entities": entities
    })

@mcp.tool()
async def semantic_search(
        document_id: str,
        query: str,
        top_k: int = 5
) -> str:
    """
    Perform semantic search on document content.

    Args:
        document_id: Document to search
        query: Semantic query to match
        top_k: Number of top results to return

    Returns:
        Most semantically relevant chunks
    """
    if document_id not in DOCUMENTS:
        return json.dumps({"error": f"Document {document_id} not found"})

    # Simple keyword-based semantic matching
    query_terms = set(query.lower().split())
    doc = DOCUMENTS[document_id]

    results = []

    for page_num, content in doc["pages"].items():
        content_lower = content.lower()

        # Calculate simple relevance score
        score = 0
        for term in query_terms:
            if len(term) > 3:  # Ignore short words
                score += content_lower.count(term) * len(term)

        if score > 0:
            # Extract first 500 chars as snippet
            snippet = content[:500].replace('\n', ' ')
            if len(content) > 500:
                snippet += "..."

            results.append({
                "page": page_num,
                "relevance_score": score,
                "snippet": snippet,
                "full_content_length": len(content)
            })

    # Sort by relevance and take top_k
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = results[:top_k]

    return json.dumps({
        "document_id": document_id,
        "query": query,
        "results_found": len(results),
        "top_results": results
    })

def extract_closing_date(text: str) -> Optional[str]:
    """Extract closing date from text"""
    patterns = [
        r'(\w+ \d{1,2}, \d{4}) \(the "Closing Date"\)',
        r'Closing Date.*?(\w+ \d{1,2}, \d{4})',
        r'dated as of (\w+ \d{1,2}, \d{4})'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def extract_context(text: str, target: str, context_size: int = 100) -> str:
    """Extract context around target string"""
    idx = text.find(target)
    if idx == -1:
        return target

    start = max(0, idx - context_size)
    end = min(len(text), idx + len(target) + context_size)

    context = text[start:end]
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context

if __name__ == "__main__":
    # Run MCP server
    mcp.run()