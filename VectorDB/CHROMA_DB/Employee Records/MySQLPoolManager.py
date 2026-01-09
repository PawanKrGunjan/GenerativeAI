# MySQLPoolManager.py - COMPLETE MERGED VERSION
import threading
from typing import Any, Dict, List, Optional
import mysql.connector.pooling as pooling
from mysql.connector import Error
from dataclasses import dataclass
import re
from chromadb.utils import embedding_functions

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

class MySQLPoolManager:
    _instance: "MySQLPoolManager" = None
    _lock = threading.Lock()
    
    pool: pooling.MySQLConnectionPool = None
    config: Dict[str, Any] = None
    _initialized = False

    def __new__(cls, config: Dict[str, Any] = None) -> "MySQLPoolManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any] = None):
        if not self._initialized and config:
            self.config = config
            self._init_pool(self.config)
            self._create_tables_if_not_exists()
            self._initialized = True
            print("✅ MySQLPoolManager initialized with pooling.")

    def _init_pool(self, config: Dict[str, Any]):
        """Initialize MySQL connection pool (YOUR ORIGINAL METHOD)"""
        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="employee_pool",  # Changed from anpr_pool
                pool_size=10,
                pool_reset_session=True,
                host=config['HOST'],
                port=config.get('PORT', 3306),
                database=config['NAME'],
                user=config['USER'],
                password=config['PASSWORD']
            )
            print("✅ MySQL connection pool created successfully.")
        except Exception as e:
            print(f"❌ Database pool initialization failed: {str(e)}")
            raise

    def _create_tables_if_not_exists(self):
        """Create employees table if not exists (YOUR CODE)"""
        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    Id VARCHAR(20) PRIMARY KEY,
                    FullName VARCHAR(100),
                    Experience INT,
                    Department VARCHAR(50),
                    Role VARCHAR(100),
                    Skills TEXT,
                    Location VARCHAR(50),
                    Salary INT,
                    Hire_date DATE
                )
            """)
            conn.commit()
            print("✅ Employees table verified/created.")
        except Error as e:
            print(f"❌ Table creation failed: {e}")
        finally:
            cursor.close()
            conn.close()

    def get_connection(self):
        """Get connection from pool"""
        if not self.pool:
            raise RuntimeError("Pool not initialized")
        return self.pool.get_connection()

    def execute_query(self, query: str, params=None, fetch=False, dictionary=False):
        """Enhanced execute_query with dictionary support for ChromaDB"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if dictionary:
                cursor = conn.cursor(dictionary=True)
            else:
                cursor = conn.cursor()
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            conn.commit()
            return True
        except Error as e:
            print(f"❌ Query failed: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def execute_many(self, query: str, data_list: list):
        """Bulk insert (YOUR ORIGINAL CODE)"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.executemany(query, data_list)
            conn.commit()
            print(f"✅ Bulk inserted {cursor.rowcount} records!")
            return True
        except Error as e:
            print(f"❌ Bulk insert failed: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def insert_sample_employees(self):
        """Insert 15 sample employees (YOUR ORIGINAL CODE)"""
        employees_data = [
            ('EMP001', 'Pawan Kumar Gunjan', 4, 'AI/ML', 'Software Engineer', 'Python,NLP,Generative AI,Databases', 'Noida', 1200000, '2022-03-15'),
            ('EMP002', 'Ravi Sharma', 8, 'Engineering', 'Senior Developer', 'Python,Django,React,AWS,Docker', 'Bengaluru', 2200000, '2018-07-22'),
            ('EMP003', 'Priya Gupta', 6, 'Data Science', 'Data Engineer', 'Python,SQL,Pandas,MLflow,TensorFlow', 'Hyderabad', 1800000, '2020-01-10'),
            ('EMP004', 'Amit Patel', 12, 'DevOps', 'DevOps Architect', 'Kubernetes,Terraform,CI/CD,Jenkins', 'Pune', 2800000, '2014-05-05'),
            ('EMP005', 'Neha Singh', 3, 'AI/ML', 'ML Engineer', 'Python,PyTorch,Computer Vision,NLP', 'Delhi', 950000, '2023-02-18'),
            ('EMP006', 'Vikram Kumar', 10, 'Engineering', 'Tech Lead', 'Java,Spring Boot,Microservices,AWS', 'Noida', 2600000, '2016-09-12'),
            ('EMP007', 'Anita Rao', 7, 'Data Science', 'Senior Analyst', 'Python,R,Tableau,PowerBI,SQL', 'Mumbai', 2000000, '2019-04-30'),
            ('EMP008', 'Suresh Reddy', 5, 'DevOps', 'SRE Engineer', 'Linux,AWS,GCP,Prometheus,Grafana', 'Hyderabad', 1600000, '2021-08-25'),
            ('EMP009', 'Kavya Menon', 9, 'Product', 'Product Manager', 'Agile,Jira,Analytics,SQL,Stakeholder Mgmt', 'Bengaluru', 2400000, '2017-11-03'),
            ('EMP010', 'Rahul Verma', 2, 'Engineering', 'Junior Developer', 'Python,JavaScript,React,PostgreSQL', 'Noida', 750000, '2024-01-20'),
            ('EMP011', 'Deepika Joshi', 11, 'AI/ML', 'AI Research Lead', 'TensorFlow,Transformers,LLM Fine-tuning', 'Gurugram', 3200000, '2015-03-10'),
            ('EMP012', 'Arjun Malik', 1, 'Data Science', 'Data Analyst', 'Excel,SQL,PowerBI,Python Basics', 'Chennai', 650000, '2025-01-05'),
            ('EMP013', 'Meera Nair', 15, 'Engineering', 'Principal Engineer', 'Go,Microservices,gRPC,Kafka', 'Bengaluru', 4500000, '2011-06-15'),
            ('EMP014', 'Karan Singh', 6, 'DevOps', 'Cloud Engineer', 'AWS,Terraform,CloudFormation,ELK', 'Mumbai', 1900000, '2020-09-01'),
            ('EMP015', 'Sneha Pillai', 4, 'Product', 'UX Researcher', 'Figma,Miro,User Testing,Analytics', 'Pune', 1100000, '2022-07-20')
        ]
        
        insert_query = """
            INSERT IGNORE INTO employees (Id, FullName, Experience, Department, Role, Skills, Location, Salary, Hire_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE Skills=VALUES(Skills)
        """
        return self.execute_many(insert_query, employees_data)

    def fetch_employees(self) -> List[Employee]:
        """Fetch employees for ChromaDB (NEW)"""
        results = self.execute_query(
            "SELECT * FROM employees ORDER BY Experience DESC", 
            fetch=True, 
            dictionary=True
        )
        
        if not results:
            return []
        
        employees = []
        for row in results:
            emp = Employee(
                id=row['Id'],
                full_name=row['FullName'],
                experience=row['Experience'],
                department=row['Department'],
                role=row['Role'],
                skills=row['Skills'],
                location=row['Location'],
                salary=row['Salary'],
                hire_date=row['Hire_date']
            )
            employees.append(emp)
        
        print(f"✅ Fetched {len(employees)} employees for ChromaDB")
        return employees

    def view_all_employees(self):
        """View all employees (YOUR ORIGINAL CODE)"""
        results = self.execute_query("SELECT * FROM employees ORDER BY Experience DESC", fetch=True)
        if results:
            print("\n📋 All Employees:")
            print("-" * 140)
            print(f"{'ID':<8} | {'FullName':<20} | {'Exp':<3} | {'Dept':<12} | {'Role':<20} | {'Location':<12} | {'Salary':<12} | {'Hire Date'}")
            print("-" * 140)
            for emp in results:
                print(f"{emp[0]:<8} | {emp[1]:<20} | {emp[2]}y | {emp[3]:<12} | {emp[4]:<20} | {emp[6]:<12} | ₹{emp[7]:,} | {emp[8]}")
            print("-" * 140)
            print(f"📊 Total: {len(results)} employees")

    def close_pool(self):
        """Close pool"""
        if self.pool:
            self.pool._remove_connections()
            print("🔌 Connection pool closed.")
