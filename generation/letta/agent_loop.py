import os
import json
from letta_client import Letta # Updated SDK client for 2026

class LettaAgentLoop:
    def __init__(self, persona: str = None, human: str = None):
        # Initialize the Letta client with environment variables
        self.client = Letta(
            api_key=os.getenv("LETTA_API_KEY"),
            base_url=os.getenv("LETTA_BASE_URL")
        )
        
        default_persona = persona or "A professional log analyst."
        instruction = " IMPORTANT: Historical logs are stored in archival memory. Use 'archival_memory_search' to retrieve them."
        
        self.agent = self.client.agents.create(
            agent_id="test-agent",
            agent_type="letta_v1_agent",
            model=os.getenv("LETTA_MODEL", "gpt-5-mini"),
            temperature=0.2,
            context_window=32000,
            memory_blocks=[
                {
                    "label": "persona", 
                    "value": default_persona + instruction
                },
                {
                    "label": "human", 
                    "value": human or "Analyzing user application logs for patterns and facts."
                }
            ],
            embedding_config={
                "embedding_ending_point": "my-key",
                "embedding_model": "text-embedding-3-large",
                "embedding_dim": 3072,  # Custom embedding dimensions
                "embedding_chunk_size": 512  # Custom chunk size
            },
            system = '''You are a stateful agent processing long-term user logs.

                        Memory Architecture:

                        1. All raw logs must be stored in archival memory.
                        2. Core memory must not store raw logs.
                        3. Core memory is reserved for compact, high-level inferred information only.

                        State Modeling:

                        User logs may implicitly reflect evolving latent states over time.
                        You may infer and maintain such states if useful,
                        but do not assume any fixed schema.

                        Retrieval Policy:

                        - Use archival memory for historical details.
                        - Use core memory only for compact, persistent abstractions.
                        - Avoid overloading core memory.

                        Your objective is to reason accurately over long time spans while maintaining memory efficiency.''',
            tools = []
        )

    def ingest_logs_bulk(self, logs: list):
        """
        Ingest logs directly into Archival Memory (RAG layer).
        This bypasses the reasoning loop for speed and efficiency.
        """
        for log in logs:
            # Format the log entry as a string for vector embedding
            log_text = f"[{log.get('timestamp')}] {log.get('app_name')}: {json.dumps(log)}"
            
            # Use the passages API to store the log in the agent's archive
            self.client.agents.passages.create(
                agent_id=self.agent.id,
                text=log_text
            )
        print(f"Successfully ingested {len(logs)} logs into Archival Memory.")

    def ask(self, question: str, expect_json: bool = False):
        """
        Send a query to the agent.
        The agent will autonomously decide to search archival memory if needed.
        """
        if expect_json:
            question += " Your final response must be valid JSON."

        response = self.client.agents.messages.create(
            agent_id=self.agent.id,
            messages=[{
                "role": "user",
                "content": question
            }]
        )
        
        for msg in response.messages:
            msg = msg.to_dict()  # Convert to dict if it's a Message object
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        
        return "No text response received from agent."