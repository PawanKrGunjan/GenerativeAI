# main.py - Application entry point and initialization
import sys
from chroma_vector_db import ChromaEmployeeVectorDB, Employee, Colors
from MySQLPoolManager import MySQLPoolManager

DB_CONFIG = {
    'HOST': 'localhost',
    'PORT': 3306,
    'NAME': 'Gunjan_EmpDB',
    'USER': 'Om',
    'PASSWORD': 'Gunjan123'
}

def initialize_vector_db(mysql_pool) -> bool:
    try:
        employees_raw = mysql_pool.fetch_employees()
        if not employees_raw:
            print(f"{Colors.WARNING}⚠️  No data found. Inserting sample employees...{Colors.END}")
            mysql_pool.insert_sample_employees()
            employees_raw = mysql_pool.fetch_employees()
        
        if not employees_raw:
            print(f"{Colors.FAIL}❌ Failed to load any employee data.{Colors.END}")
            return False
        
        employees = [Employee(**emp.__dict__) for emp in employees_raw]
        
        vector_db = ChromaEmployeeVectorDB()
        count = vector_db.add_employees(employees)
        
        print(f"{Colors.OKGREEN}\n✅ System ready! {count} employees indexed in ChromaDB.{Colors.END}")
        return vector_db
    except Exception as e:
        print(f"{Colors.FAIL}❌ Initialization failed: {e}{Colors.END}")
        return False

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}🚀 Starting Employee Vector Search System...{Colors.END}")
    
    mysql_pool = MySQLPoolManager(DB_CONFIG)
    vector_db = initialize_vector_db(mysql_pool)
    
    if not vector_db:
        print(f"{Colors.FAIL}❌ Cannot start application due to initialization error.{Colors.END}")
        return 1
    
    # Launch interactive session
    vector_db.interactive_search()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())