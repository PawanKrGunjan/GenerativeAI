
# Employee Smart Search System 🚀

A **user-friendly, semantic search tool** for finding employees using natural language queries combined with precise filters. Built with **ChromaDB** for vector-based semantic search and connected to a MySQL database for data source.

Perfect for HR teams, managers, or anyone who wants to quickly find the right person without complex queries!

## Features ✨

- **Natural Language Search**  
  Just type in plain English!  
  Examples:  
  - `Python developer in Bengaluru`  
  - `Senior AI engineer with TensorFlow`  
  - `People in Noida with more than 5 years experience`

- **Precise Filters**  
  Narrow down results by:  
  - Department (e.g., AI/ML, Engineering)  
  - Location (e.g., Noida, Bengaluru, Pune)  
  - Role  
  - Experience range  
  - Salary range  
  - Specific skills (e.g., Python, AWS)

- **Beautiful Terminal Interface**  
  Colorful, easy-to-read results with match percentages

- **Useful Commands**  
  - `help` → Full user guide  
  - `stats` → View total employees, average salary, and department breakdown  
  - `save` → Export all employee data to a JSON file  
  - `list-all` → Show all employees  
  - `clear` → Clear screen  
  - `quit` → Exit

- **Persistent Storage**  
  Data is saved automatically using ChromaDB (no loss on restart)

## Project Structure

Employee-Smart-Search/
├── main.py                      # Application entry point & startup
├── chroma_vector_db.py          # Core ChromaDB logic + interactive search UI
├── MySQLPoolManager.py          # MySQL connection handler (provided separately)
├── chroma_db/                   # Auto-created: Contains chroma.sqlite3 (database)
├── employees_export.json        # Created when you use 'save' command
└── README.md                    # This file


## Requirements

- Python 3.8+
- ChromaDB (`chromadb`)
- Sentence Transformers (`sentence-transformers`)
- MySQL server running locally or accessible

Install dependencies:

```bash
pip install chromadb sentence-transformers
```

> Note: No additional packages needed for MySQL if using `mysql-connector-python` or similar (included in your `MySQLPoolManager`).

## Setup & Database Configuration

1. Make sure your MySQL server is running.
2. Update the database config in `main.py` if needed:

```python
DB_CONFIG = {
    'HOST': 'localhost',
    'PORT': 3306,
    'NAME': 'Gunjan_EmpDB',      # Your database name
    'USER': '**',               # Your MySQL username
    'PASSWORD': '*********'     # Your MySQL password
}
```

3. Ensure your MySQL database has an `employees` table (or the script will insert sample data on first run).

## How to Run

```bash
python main.py
```

On first launch:
- If no data exists → Sample employees will be inserted automatically
- All data will be indexed into ChromaDB for fast semantic search

Then enjoy the interactive search!

## Example Usage in App

```
Search or command: Senior developer in Engineering

How many results to show? (default 10): 5

Optional Filters:
   Department: Engineering
   Location: 
   Min Experience: 8

→ Shows top 5 senior developers in Engineering with 8+ years experience
```

## Data Storage

- All vector + metadata stored in:  
  `./chroma_db/chroma.sqlite3`
- This file persists across sessions
- Use `save` command to export readable JSON anytime

## Sample Stats Output

```
EMPLOYEE DATABASE SUMMARY
==================================================
Total Employees: 15
Average Salary: ₹1,976,666
==================================================
Employees by Department:
   • Engineering     : 4 employees
   • DevOps          : 3 employees
   • AI/ML           : 3 employees
   • Data Science    : 3 employees
   • Product         : 2 employees
==================================================
```

## For Developers

Want to extend this project?
- Add CSV export
- Add web interface (Streamlit/FastAPI)
- Connect to HR systems
- Add resume upload & parsing

Contributions welcome! 🌟

## Made With ❤️ By

Pawan Kumar Gunjan  
January 2026

Happy Searching! 🔍
