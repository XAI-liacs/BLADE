from blade.experiment import MA_BBOB_Experiment
from blade.llm import Gemini_LLM
from blade.methods import LLaMEA, RandomSearch
import os

api_key = os.getenv("GEMINI_API_KEY")
ai_model = "gemini-2.0-flash"
llm = Gemini_LLM(api_key, ai_model)
budget = 10
RS = RandomSearch(llm, budget=10) #LLaMEA(llm)
LLaMEA_method = LLaMEA(llm, budget=10)

experiment = MA_BBOB_Experiment(methods=[LLaMEA_method], llm=llm, runs=1, dims=[2], budget_factor=100) #quick run
print(experiment())

