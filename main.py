import sys
import os

sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
)
import uvicorn

def main():
    uvicorn.run("src.api.api:app", reload=True, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
