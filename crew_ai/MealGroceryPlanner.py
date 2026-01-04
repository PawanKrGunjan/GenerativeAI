from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from IPython.display import display, JSON, Markdown
import os
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from dotenv import load_dotenv
from leftover import LeftoversCrew

# Set environment variables
load_dotenv()

perplexity_llm = LLM(
    model="sonar-pro", 
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY"),  # Required!
    # temperature=0.3,
    # max_tokens=2000
)
ollama_llm = LLM(
    model="ollama/llama3.2:latest",
    base_url="http://localhost:11434",
    max_tokens= 500
)

class GroceryItem(BaseModel):
    """Individual grocery item"""
    name: str = Field(description="Name of the grocery item")
    quantity: str = Field(description="Quantity needed (for example, '2 kg', '200 gram')")
    estimated_price: str = Field(description="Estimated price (for example, '₹100-500')")
    category: str = Field(description="Store section (for example, 'Produce', 'Dairy')")



class MealPlan(BaseModel):
    """Simple meal plan"""
    meal_name: str = Field(description="Name of the meal")
    taste: str = Field(description="Taste profile of the meal 'Salty', 'Sweet', 'Sour'")
    servings: int = Field(description="Number of people it serves")
    researched_ingredients: List[str] = Field(description="Ingredients found through research")



class ShoppingCategory(BaseModel):
    """Store section with items"""
    section_name: str = Field(description="Store section (for example, 'Produce', 'Dairy')")
    items: List[GroceryItem] = Field(description="Items in this section")
    estimated_total: str = Field(description="Estimated cost for this section in Indian Rupees (₹)")


class GroceryShoppingPlan(BaseModel):
    """Complete simplified shopping plan"""
    total_budget: str = Field(description="Total planned budget")
    meal_plans: List[MealPlan] = Field(description="Planned meals")
    shopping_sections: List[ShoppingCategory] = Field(description="Organized by store sections")
    shopping_tips: List[str] = Field(description="Money-saving and efficiency tips")


meal_planner = Agent(
    role="Meal Planner & Recipe Researcher",
    goal="Search for optimal recipes and create detailed meal plans",
    backstory="A skilled meal planner who researches the best recipes online, considering dietary needs, taste preferences, and budget constraints.",
    tools=[SerperDevTool()],
    llm=perplexity_llm,
    verbose=False)

meal_planning_task = Task(
    description=(
        "Search for the best '{meal_name}' recipe for {servings} people within a {budget} budget. "
        "Consider dietary restrictions: {dietary_restrictions} and Taste preference : {taste}. "
        "Find recipes that match the Taste preference and provide complete ingredient lists with quantities."
    ),
    expected_output="A detailed meal plan with researched ingredients, quantities, and cooking instructions appropriate for the skill level.",
    agent=meal_planner,
    output_pydantic=MealPlan,
    output_file="meals.json"
)


from crewai import Crew, Process

meal_planner_crew = Crew(
    agents=[meal_planner],
    tasks=[meal_planning_task],
    process=Process.sequential,  # Ensures tasks are executed in order
    verbose=True
)

meal_planner_result = meal_planner_crew.kickoff(
    inputs={
        "meal_name": "Kheer",
        "servings": 4,
        "budget": "₹25",                           
        "dietary_restrictions": ["no nuts"],       
        "taste": "Sweet"                
    }
)

print('-'*100)
print("✅ Single meal planning completed!")
print("📋 Single Meal Results:")
print(meal_planner_result)
print('-'*100)
#----------------------------------------------------------------------------------------------------------------
shopping_organizer = Agent(
    role="Shopping Organizer", 
    goal="Organize grocery lists by store sections efficiently",
    backstory="An experienced shopper who knows how to organize lists for quick store trips and considers dietary restrictions.",
    tools=[],
    llm=perplexity_llm,
    verbose=False
)

# shopping_task = Task(
#     description=(
#         "Organize the ingredients from the '{meal_name}' meal plan into a grocery shopping list. "
#         "Group items by store sections and estimate quantities for {servings} people. "
#         "Consider dietary restrictions: {dietary_restrictions} and Taste Preferences: {taste}. "
#         "Stay within budget: {budget}."
#     ),
#     expected_output="An organized shopping list grouped by store sections with quantities and prices.",
#     agent=shopping_organizer,
#     context=[meal_planning_task],
#     output_pydantic=GroceryShoppingPlan,
#     output_file="shopping_list.json"
# )

shopping_task = Task(
    description=(
        "Using the meal plan and ingredient list from the previous task, "
        "create a complete grocery shopping plan for '{meal_name}' serving {servings} people. "
        "Organize all required ingredients by store sections (e.g., Dairy, Pantry, Produce). "
        "Estimate realistic quantities and price ranges in Indian Rupees based on current market rates. "
        "Total budget must not exceed {budget}. "
        "Dietary restrictions: {dietary_restrictions}. Taste: {taste}."
    ),
    expected_output="A complete grocery shopping plan with sections, items, quantities, estimated prices, and total budget.",
    agent=shopping_organizer,
    context=[meal_planning_task],  # This passes previous output
    output_pydantic=GroceryShoppingPlan,  # ← Keep this ONLY here
    output_file="shopping_list.json"
)

two_agent_grocery_crew = Crew(
    agents=[meal_planner, shopping_organizer],  # Both agents
    tasks=[meal_planning_task, shopping_task],   # Both tasks
    process=Process.sequential,
    verbose=True,
    memory = False,
)

# Run the complete crew (this will do BOTH meal planning AND shopping)
shopping_result = two_agent_grocery_crew.kickoff(
    inputs={
        "meal_name": "Kheer",
        "servings": 4,
        "budget": "₹25",                           
        "dietary_restrictions": ["no nuts"],      
        "taste": "Sweet"               
    }
)
print('-'*100)
# Print the shopping results
print("✅ Complete meal planning + shopping completed!")
print("📋 Shopping Results:")
print(shopping_result)

print('-'*100)

#-----------------------------------------------------------------------------------------------------

budget_advisor = Agent(
    role="Budget Advisor",
    goal="Provide cost estimates and money-saving tips",
    backstory="A budget-conscious shopper who helps families save money on groceries while respecting dietary needs.",
    tools=[SerperDevTool()],
    llm=perplexity_llm,
    verbose=False
)

budget_task = Task(
    description=(
        "Analyze the shopping plan for '{meal_name}' serving {servings} people. "
        "Ensure total cost stays within {budget}. Consider dietary restrictions: {dietary_restrictions}. "
        "Provide practical money-saving tips and alternative ingredients if needed to meet budget."
    ),
    expected_output="A complete shopping guide with detailed prices, budget analysis, and money-saving tips.",
    agent=budget_advisor,
    context=[meal_planning_task, shopping_task],
    output_file="shopping_guide.md"
)


leftovers_cb = LeftoversCrew(llm=perplexity_llm)
yaml_leftover_manager = leftovers_cb.leftover_manager()
yaml_leftover_task    = leftovers_cb.leftover_task()

summary_agent = Agent(
    role="Report Compiler",
    goal="Compile comprehensive meal planning reports from all team outputs",
    backstory="A skilled coordinator who organizes information from multiple specialists into comprehensive, easy-to-follow reports.",
    tools=[],
    llm=perplexity_llm,
    verbose=False
)

summary_task = Task(
    description=(
        "Compile a comprehensive meal planning report that includes:\n"
        "1. The complete recipe and cooking instructions from the meal planner\n"
        "2. The organized shopping list with prices from the shopping organizer\n"
        "3. The budget analysis and money-saving tips from the budget advisor\n"
        "4. The leftover management suggestions from the waste reduction specialist\n"
        "Format this as a complete, user-friendly meal planning guide."
    ),
    expected_output="A comprehensive meal planning guide that combines all team outputs into one cohesive report.",
    agent=summary_agent,
    context=[meal_planning_task, shopping_task, budget_task, yaml_leftover_task],
)

complete_grocery_crew = Crew(
    agents=[
        meal_planner,           
        shopping_organizer,      
        budget_advisor,         
        yaml_leftover_manager,  # YAML-based leftover manager
        summary_agent           # New summary agent
    ],
    tasks=[
        meal_planning_task,     
        shopping_task,          
        budget_task,            
        yaml_leftover_task,    # YAML-based leftover task
        summary_task            # New summary task
    ],
    process=Process.sequential,
    verbose=True
)

# Run the complete crew
complete_result = complete_grocery_crew.kickoff(
    inputs={
        "meal_name": "Kheer",
        "servings": 4,
        "budget": "₹25",
        "dietary_restrictions": ["no nuts", "low sodium"],
        "taste": "sweet"
    }
)
print('-'*100)
print("✅ Complete meal planning with summary completed!")
print("📋 Complete Results:")
print(complete_result)

print('-'*100)

#-----------------------------------------------------------------------------------------------------
