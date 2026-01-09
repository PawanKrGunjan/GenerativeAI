# chroma_vector_db.py - Core ChromaDB + Enhanced Interactive UI
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dataclasses import dataclass
import os
import sys
import platform
from collections import Counter

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Colors for beautiful terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def clear_screen():
    os.system('cls' if platform.system() == "Windows" else 'clear')

@dataclass
class Employee:
    id: str
    full_name: str
    experience: int
    department: str
    role: str
    skills: str
    location: str
    salary: int
    hire_date: str
    
    def to_document(self) -> str:
        return (
            f"{self.role} with {self.experience} years experience in {self.department}. "
            f"Skills: {self.skills}. Location: {self.location}. "
            f"Salary: ₹{self.salary:,}. Name: {self.full_name}"
        )
    
    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": str(self.full_name),
            "id": str(self.id),
            "department": str(self.department),
            "role": str(self.role),
            "experience": int(self.experience),
            "location": str(self.location),
            "salary": int(self.salary),
            "skills": str(self.skills)[:200],
            "hire_date": str(self.hire_date)
        }

import logging
from pathlib import Path
from typing import List
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class ChromaEmployeeVectorDB:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "employee_collection"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.collection_name = collection_name
        
        # Best lightweight model for fast, accurate semantic search
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Persistent client — saves data to disk
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Create or load collection with optimal settings
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """
        Safely get or create the Chroma collection with best practices.
        
        This is the RECOMMENDED approach for most use cases (including yours).
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Employee semantic search database",
                "hnsw:space": "cosine"  # Explicitly use cosine similarity — best for text embeddings
            }
        )
    
    # ===================================================================
    # NOTES: When to use which collection creation method?
    # ===================================================================
    #
    # 1. RECOMMENDED (Use this — what we have above):
    #    get_or_create_collection() + embedding_function + "hnsw:space": "cosine" in metadata
    #
    #    Why?
    #    - Safe: Won't fail if collection already exists
    #    - Clean and readable
    #    - Automatically uses cosine distance (ideal for sentence transformers)
    #    - Explicit control via metadata
    #
    # 2. Alternative: Separate get + create (your original code)
    #    try: get_collection() except: create_collection(...)
    #
    #    Why use?
    #    - Only if you want custom logic when collection exists vs. when creating new
    #    - Slightly more control, but more verbose
    #
    # 3. Advanced: Force different distance metric (rarely needed)
    #    metadata={"hnsw:space": "l2"} or "ip"
    #
    #    When?
    #    - Only if using custom embeddings NOT based on sentence transformers
    #    - Cosine is almost always best for natural language / semantic search
    #
    # 4. NEVER do this (Invalid):
    #    configuration={"hnsw": {...}, "embedding_function": ...}
    #    → No such parameter exists in current ChromaDB versions
    #
    # ===================================================================

    def add_employees(self, employees: List[Employee]) -> int:
        """
        Add or update employees in the vector database.
        Uses upsert behavior via .add() — safe for re-indexing.
        """
        try:
            docs = [emp.to_document() for emp in employees]
            ids = [emp.id for emp in employees]
            metadatas = [emp.to_metadata() for emp in employees]
            
            # .add() automatically upserts (updates if ID exists)
            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metadatas
            )
            
            total_count = self.collection.count()
            logger.info(f"Successfully indexed {len(employees)} employees. Total: {total_count}")
            return total_count
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            logger.info("Attempting to recreate collection and re-index...")
            self._recreate_and_index(employees)
            return len(employees)
    
    def _recreate_and_index(self, employees: List[Employee]):
        """Force recreate collection — use only as fallback on corruption"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Old collection deleted")
        except:
            pass  # Ignore if already gone
        
        # Recreate with same optimal settings
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Employee semantic search database",
                "hnsw:space": "cosine"
            }
        )
        
        docs = [emp.to_document() for emp in employees]
        ids = [emp.id for emp in employees]
        metadatas = [emp.to_metadata() for emp in employees]
        
        self.collection.add(ids=ids, documents=docs, metadatas=metadatas)
        logger.info("Collection recreated and employees re-indexed")

    def build_where_filter(self, dept=None, loc=None, role=None, min_exp=None, max_exp=None,
                           min_sal=None, max_sal=None, skills_contains=None) -> Optional[Dict]:
        filters = []
        if dept: filters.append({"department": dept})
        if loc: filters.append({"location": loc})
        if role: filters.append({"role": role})
        if min_exp:
            try: filters.append({"experience": {"$gte": int(min_exp)}})
            except: print(f"{Colors.FAIL}Invalid minimum experience{Colors.END}")
        if max_exp:
            try: filters.append({"experience": {"$lte": int(max_exp)}})
            except: print(f"{Colors.FAIL}Invalid maximum experience{Colors.END}")
        if min_sal:
            try: filters.append({"salary": {"$gte": int(min_sal)}})
            except: print(f"{Colors.FAIL}Invalid minimum salary{Colors.END}")
        if max_sal:
            try: filters.append({"salary": {"$lte": int(max_sal)}})
            except: print(f"{Colors.FAIL}Invalid maximum salary{Colors.END}")
        if skills_contains:
            filters.append({"skills": {"$contains": skills_contains}})
        
        if not filters: return None
        return filters[0] if len(filters) == 1 else {"$and": filters}

    def similarity_search(self, query: str, n_results: int = 10, where: Optional[Dict] = None) -> List[Dict]:
        try:
            results = self.collection.query(
                query_texts=[query or "employee"],
                n_results=n_results,
                where=where,
                include=["metadatas", "distances"]
            )
            return self._format_results(results)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            try:
                results = self.collection.query(
                    query_texts=[query or "employee"],
                    n_results=n_results,
                    include=["metadatas", "distances"]
                )
                return self._format_results(results)
            except:
                return []

    def _format_results(self, results) -> List[Dict]:
        formatted = []
        if results['ids'][0]:
            for i, (doc_id, meta, dist) in enumerate(zip(results['ids'][0], results['metadatas'][0], results['distances'][0])):
                formatted.append({
                    "rank": i + 1,
                    "id": doc_id,
                    "name": meta.get("name", "Unknown"),
                    "role": meta.get("role", "Unknown"),
                    "department": meta.get("department", "Unknown"),
                    "experience": meta.get("experience", 0),
                    "location": meta.get("location", "Unknown"),
                    "salary": meta.get("salary", 0),
                    "similarity": 1.0 - float(dist)
                })
        return formatted

    def print_results(self, results: List[Dict]):
        if not results:
            print(f"{Colors.FAIL}\nNo employees found matching your search.{Colors.END}")
            return
        
        print(f"{Colors.OKGREEN}\nTOP {len(results)} BEST MATCHES:{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 150}{Colors.END}")
        print(f"{Colors.BOLD}{'#' :<3} {'Name' :<22} {'Role' :<22} {'Department' :<12} {'Exp' :<4} {'City' :<12} {'Salary' :<12} {'Match %' :<7}{Colors.END}")
        print(f"{Colors.HEADER}{'-' * 150}{Colors.END}")
        for r in results:
            sim_pct = r['similarity'] * 100
            print(f"{r['rank'] :<3} {r['name'][:21] :<22} {r['role'][:21] :<22} "
                  f"{r['department'][:11] :<12} {r['experience'] :<4}y "
                  f"{r['location'][:11] :<12} ₹{r['salary']:>10,} {sim_pct:>6.1f}%")
        print(f"{Colors.HEADER}{'=' * 150}{Colors.END}")

    def print_enhanced_stats(self):
        try:
            data = self.collection.get(include=["metadatas"])
            metadatas = data["metadatas"]
            total = len(metadatas)
            
            if total == 0:
                print(f"{Colors.WARNING}No employee data available yet.{Colors.END}")
                return
            
            departments = [m["department"] for m in metadatas]
            dept_count = Counter(departments)
            salaries = [m["salary"] for m in metadatas]
            avg_salary = sum(salaries) // total
            
            print(f"{Colors.OKGREEN}\nEMPLOYEE DATABASE SUMMARY{Colors.END}")
            print(f"{Colors.HEADER}{'=' * 50}{Colors.END}")
            print(f"{Colors.BOLD}Total Employees:{Colors.END} {total}")
            print(f"{Colors.BOLD}Average Salary:{Colors.END}  ₹{avg_salary:,}")
            print(f"{Colors.HEADER}{'=' * 50}{Colors.END}")
            print(f"{Colors.BOLD}Employees by Department:{Colors.END}")
            for dept, count in sorted(dept_count.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {dept:<15} : {count} employee{'s' if count != 1 else ''}")
            print(f"{Colors.HEADER}{'=' * 50}{Colors.END}")
            print(f"{Colors.OKBLUE}Data stored at: {self.persist_dir.absolute()}{Colors.END}")
        except Exception as e:
            print(f"{Colors.FAIL}Could not load stats: {e}{Colors.END}")

    def export_to_json(self, filename: str = "employees_export.json"):
        try:
            # FIXED: Only request metadatas and ids
            data = self.collection.get(include=["metadatas"])
            export_data = []
            all_ids = data.get("ids", [])
            for i, meta in enumerate(data["metadatas"]):
                record = {"id": all_ids[i] if i < len(all_ids) else f"emp_{i}"}
                record.update(meta)
                export_data.append(record)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"{Colors.OKGREEN}\nSuccessfully exported {len(export_data)} employees to '{filename}' 🎉{Colors.END}")
            print(f"{Colors.OKBLUE}You can now open this file in Excel or any text editor.{Colors.END}")
        except Exception as e:
            print(f"{Colors.FAIL}Export failed: {e}{Colors.END}")

    def print_help(self):
        print(f"{Colors.WARNING}\nHOW TO USE THIS EMPLOYEE SEARCH TOOL{Colors.END}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.END}")
        print(f"{Colors.OKBLUE}Just type what you're looking for in simple English!{Colors.END}")
        print(f"   Examples:")
        print(f"   • Python developer in Bengaluru")
        print(f"   • Senior manager")
        print(f"   • Data scientist with AWS")
        print(f"   • People in Noida")
        print(f"\n{Colors.OKBLUE}After typing your search, you can add filters like:{Colors.END}")
        print(f"   • Department → e.g., AI/ML, Engineering")
        print(f"   • Location    → e.g., Bengaluru, Pune")
        print(f"   • Minimum Experience, Salary, etc.")
        print(f"\n{Colors.OKGREEN}Useful Commands:{Colors.END}")
        print(f"   help      → Show this guide again")
        print(f"   stats     → See total employees & department breakdown")
        print(f"   save      → Save all employee data to a file (JSON)")
        print(f"   list-all  → Show all employees")
        print(f"   clear     → Clear the screen")
        print(f"   quit      → Exit the program")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.END}")

    def interactive_search(self):
        clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.OKGREEN}{Colors.BOLD}      EMPLOYEE SMART SEARCH SYSTEM{Colors.END}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.OKBLUE}Type anything like: 'Python developer in Bengaluru' or 'Senior AI engineer'{Colors.END}")
        print(f"{Colors.WARNING}Type 'help' for full guide • 'stats' for summary • 'quit' to exit{Colors.END}")
        print(f"{Colors.HEADER}{'-'*80}{Colors.END}")

        while True:
            print(f"\n{Colors.HEADER}{'-'*80}{Colors.END}")
            query = input(f"{Colors.OKGREEN}Search or command: {Colors.END}").strip()
            
            cmd = query.lower()
            if cmd in ['quit', 'exit', 'q']:
                print(f"{Colors.OKGREEN}\nThank you! Goodbye 👋{Colors.END}")
                break
            if cmd == 'help':
                self.print_help()
                continue
            if cmd == 'stats':
                self.print_enhanced_stats()
                continue
            if cmd == 'save':
                filename = input(f"{Colors.OKBLUE}Enter filename (default: employees_export.json): {Colors.END}").strip()
                if not filename.endswith('.json'):
                    filename = filename + '.json' if filename else "employees_export.json"
                self.export_to_json(filename or "employees_export.json")
                continue
            if cmd == 'list-all':
                results = self.similarity_search("", n_results=50)
                self.print_results(results)
                continue
            if cmd == 'clear':
                clear_screen()
                continue

            n_str = input(f"{Colors.OKBLUE}How many results to show? (default 10): {Colors.END}").strip()
            n_results = int(n_str) if n_str.isdigit() else 10

            print(f"\n{Colors.OKBLUE}Optional Filters (press Enter to skip each):{Colors.END}")
            dept = input("   Department (e.g., AI/ML): ").strip()
            loc = input("   Location (e.g., Bengaluru): ").strip()
            role = input("   Role (e.g., Developer): ").strip()
            min_exp = input("   Min Experience (years): ").strip()
            max_exp = input("   Max Experience (years): ").strip()
            min_sal = input("   Min Salary: ").strip()
            max_sal = input("   Max Salary: ").strip()
            skill = input("   Must have skill (e.g., Python): ").strip()

            where = self.build_where_filter(dept, loc, role, min_exp, max_exp, min_sal, max_sal, skill)

            print(f"\n{Colors.OKGREEN}Searching...{Colors.END}")
            results = self.similarity_search(query or "employee", n_results, where)
            self.print_results(results)