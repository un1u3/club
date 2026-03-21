# CLUB — Continuous Learning Understanding Bots

A local multi-agent AI study assistant for students.  
Drop your notes, PDFs, or images — CLUB reads, summarizes, quizzes, solves, and plans for you.

## Tech Stack

| Component          | Tool        |
|--------------------|-------------|
| LLM                | ollama (llama3) |
| Agent Orchestration| LangGraph   |
| Vector Memory      | ChromaDB    |
| Chat UI            | Chainlit    |
| Language           | Python      |

## Project Structure

```
club/
├── core/              # Orchestration: watcher, coordinator, memory
├── agents/            # AI agents: reader, summarizer, quizzer, solver, planner
├── knowmyschool/      # Student profile and feedback
├── interface/         # Chainlit chat UI
├── folder/            # Study material drop zone
│   ├── notes/         #   lecture notes and textbooks
│   ├── pyqs/          #   past year questions
│   ├── images/        #   photos of handwritten notes
│   ├── youtube/       #   youtube transcripts
│   └── output/        #   generated summaries, quizzes, plans
├── config.yaml        # Central configuration
├── requirements.txt   # Python dependencies
└── README.md
```

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd club

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start ollama with llama3
ollama serve
ollama pull llama3

# 5. Launch CLUB
chainlit run interface/app.py
```

## License

MIT