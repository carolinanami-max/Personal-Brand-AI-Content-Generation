import os
import re
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.primary_kb = []
        self.secondary_kb = []
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def load_markdown_files(self, directory_path: str) -> List[Dict]:
        """Load all markdown files from a directory"""
        documents = []
        
        if not os.path.exists(directory_path):
            print(f"Directory not found: {directory_path}")
            return documents
            
        for filename in os.listdir(directory_path):
            if filename.endswith('.md'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        
                        # Simple markdown to text conversion
                        text = re.sub(r'[#*_`\[\]]+', '', content)
                        
                        documents.append({
                            'content': text,
                            'filename': filename,
                            'type': 'primary' if 'primary' in directory_path else 'secondary'
                        })
                    print(f"  Loaded: {filename}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
        
        return documents
    
    def load_all(self):
        """Load both primary and secondary knowledge bases"""
        print("Loading Primary Knowledge Base...")
        self.primary_kb = self.load_markdown_files(
            os.path.join(self.base_dir, "knowledge_base", "primary")
        )
        
        print("Loading Secondary Knowledge Base...")
        self.secondary_kb = self.load_markdown_files(
            os.path.join(self.base_dir, "knowledge_base", "secondary")
        )
        
        print(f"Loaded {len(self.primary_kb)} primary and {len(self.secondary_kb)} secondary documents")
        return self.primary_kb, self.secondary_kb
    
    def search(self, query: str, top_k: int = 3) -> str:
        """Simple keyword search - returns formatted context for prompts"""
        query_words = set(query.lower().split())
        results = []
        
        # Search through all documents
        all_docs = self.primary_kb + self.secondary_kb
        
        for doc in all_docs:
            # Calculate simple relevance score
            content_lower = doc['content'].lower()
            score = 0
            for word in query_words:
                if len(word) > 3:  # Ignore short words
                    score += content_lower.count(word)
            
            if score > 0:
                results.append((score, doc))
        
        # Sort by relevance only (avoid comparing dict payloads on score ties)
        results.sort(key=lambda item: item[0], reverse=True)
        
        # Format context for prompt
        if not results:
            return "No specific context found. Use general knowledge."
        
        context = "## RELEVANT INFORMATION FROM KNOWLEDGE BASES:\n\n"
        for score, doc in results[:top_k]:
            source = "🏢 COMPANY" if doc['type'] == 'primary' else "📊 INDUSTRY"
            context += f"{source} - {doc['filename']}:\n"
            context += f"{doc['content'][:300]}...\n\n"
        
        return context
