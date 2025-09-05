from iohblade.llm import Ollama_LLM
import time

def Get_Client(model):
    ollama_instance = Ollama_LLM(model=model)
    messages = []

    def prompt_LLM(message : str) -> str:
        """
        Testing the functionality of ollama's new client implementation on local machine.
        Spawned a ollama instance of LLM active for 5 miniutes.
        """
        messages.append({"role" : 'user', "content": message})
        response = ollama_instance._query(messages, temperature=0.8)
        messages.append({"role" : ollama_instance.model, "content" : response})
        return response

    def chatLog():
        for message in messages:
            print(f'{message["role"]} : {message["content"]}')
        print("--------------------------------------------------------------------")

    return prompt_LLM, chatLog

if __name__ == "__main__":
    start_time = time.time()
    client, chatLog = Get_Client(model="qwen3:14b")
    end_time = time.time()
    print(f"Took, {end_time - start_time}s, to start the client.")

    start_time = time.time()
    print(client(message="""Write a function to find an element in an array as quickly as possible, withot using built in
        functions, respond with <description>\n<code>"""))
    end_time = time.time()
    print(f"Took, {end_time - start_time}s, for first response.")

    start_time = time.time()
    print(client(message="""Assume that the elements are sorted in ascending order, implement a faster search strategy, and be consistent with repsonse format"""))
    end_time = time.time()
    print(f"Took, {end_time - start_time}s, for second response.")

    start_time = time.time()
    print(client(message="""Assume that the elements are sorted in descending order order, implement a faster search strategy, and be consistent with repsonse format"""))
    end_time = time.time()
    print(f"Took, {end_time - start_time}s, for final response.")
    print("""--------------------------------""")
    chatLog()
