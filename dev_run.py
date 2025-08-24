import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        reload_includes=["server.py", "rag_academico_mvp_python.py"],
        reload_excludes=[
            "env_RAG/**",
            "**/env_RAG/**",
            "**/Lib/site-packages/**",
            "**/site-packages/**",
            "**/__pycache__/**",
        ],
    )
